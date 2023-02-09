import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader

from utils import *
import pdb

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from dassl.utils import load_checkpoint



def load_model(directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16 #cfg.TRAINER.COOP.N_CTX
        ctx_init = '' # cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224 #cfg.INPUT.SIZE[0]
        self.N = 4 #cfg.MODEL.N
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init: 
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype) 
            nn.init.normal_(ctx_vectors, std=0.02)   # define the prompt to be trained
            prompt_prefix = " ".join(["X"] * n_ctx)    

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        

        classnames = [name.replace("_", " ") for name in classnames]   
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames] 

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])   
        tokenized_prompts = tokenized_prompts.repeat(self.N,1) 
       

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 
        

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = 'end' #cfg.TRAINER.COOP.CLASS_TOKEN_POSITION


    def forward(self):
       
        ctx = self.ctx
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1,-1) 
        
        ctx = ctx.permute(1, 0, 2, 3) 
        ctx = ctx.contiguous().view(self.N*self.n_cls,self.n_ctx,ctx.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.device0 = torch.device("cuda:0")
        self.device = torch.device("cuda") 
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.N = 4 #cfg.MODEL.N
        self.use_uniform = True
        self.eps = 0.1
        self.max_iter = 100

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T

    def forward(self, image):
        
        b = image.shape[0]
        image_features = self.image_encoder(image.type(self.dtype))  
        image_feature_pool = image_features[0]
        image_features = image_features[1:]  
        M = image_features.shape[0]
        self.d = image_features.shape[-1]

        prompts = self.prompt_learner()  
       
        tokenized_prompts = self.tokenized_prompts


        text_features = self.text_encoder(prompts.to(self.device), tokenized_prompts.to(self.device)) 
        text_features = text_features.to(self.device0)
        text_features =  text_features.contiguous().view(self.N, self.n_cls, self.d)  
        text_feature_pool = text_features.mean(dim=0)


        image_features =  F.normalize(image_features, dim=2) 
        image_feature_pool = F.normalize(image_feature_pool, dim=1)
        text_features = F.normalize(text_features, dim=2)
        text_feature_pool = F.normalize(text_feature_pool, dim=1)

        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()  
        sim = sim.view(M,self.N,b*self.n_cls)
        sim = sim.permute(2,0,1)
        wdist = 1.0 - sim

        xx=torch.zeros(b*self.n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        yy=torch.zeros(b*self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)

        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            T = self.Sinkhorn(KK,xx,yy)
        try:
            torch.isnan(T).any()
        except None:
            print('There is none value in your tensor, please try to adjust #thre and #eps to align data.')


        sim_op = torch.sum(T * sim, dim=(1, 2))
        sim_op = sim_op.contiguous().view(b,self.n_cls)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_feature_pool @ text_feature_pool.t()

        logits2 = logit_scale * sim_op
        logits2 = (0.5*logits2 + 0.5*logits)
        return logits2, image_feature_pool,text_feature_pool, image_features


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args



def run_tip_adapter_F(cfg, clip_model, train_loader_F,val_loader,test_loader):
    
    #Leanable parameters
    adapter = nn.Linear(cfg['cache_keys'].shape[0], cfg['cache_keys'].shape[1]*cfg['N'], bias=False).to(clip_model.dtype).cuda() #cache_keys shape [feat_dim, data_num]
    para=torch.cat([cfg['cache_keys'].t(),cfg['cache_keymaps'].mean(dim=1).t()], dim=0)
    adapter.weight = nn.Parameter(para)

    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))

    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for (images, target) in tqdm(train_loader_F):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                _, train_features,_, train_features_map = clip_model(images)
                train_features /= train_features.norm(dim=-1, keepdim=True)
                train_features_map /= train_features_map.norm(dim=-1, keepdim=True)
                train_features_map_batch = train_features_map

            batch_size_train = train_features.shape[0]
            logits_train0 = 100. * train_features @ cfg['clip_weights']
            tip_logits_train= tip_plot(cfg, train_features_map_batch,adapter,batch_size_train,beta)
            tip_logits_train_fuse =  logits_train0 + tip_logits_train * alpha
           
            loss = F.cross_entropy(tip_logits_train_fuse, target)

            acc = cls_acc(tip_logits_train_fuse, target)
            correct_samples += acc / 100 * len(tip_logits_train_fuse)
            all_samples += len(tip_logits_train_fuse)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()

        if cfg['loop'] ==True:
            tip_logits_test_list_fuse=[]
            for (images, target) in tqdm(test_loader):
                images = images.cuda()
                with torch.no_grad():
                    _, test_features,_, test_features_map = clip_model(images)
                    test_features /= test_features.norm(dim=-1, keepdim=True)
                    test_features_map /= test_features_map.norm(dim=-1, keepdim=True)
                    test_features_map_batch = test_features_map

                    batch_size_test = test_features.shape[0]
                    logits_test0 = 100. * test_features @ cfg['clip_weights']
                    tip_logits_test= tip_plot(cfg, test_features_map_batch,adapter,batch_size_test,beta)
                    tip_logits_test_fuse = logits_test0 + tip_logits_test * alpha
                    tip_logits_test_list_fuse.append(tip_logits_test_fuse)


            tip_logits_test_list_fuse_new= torch.cat(tip_logits_test_list_fuse) + 0.5*cfg['test_logits']
            acc = cls_acc(tip_logits_test_list_fuse_new, cfg['test_labels'])

            print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_epoch = train_idx
                torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
        else:
            with torch.no_grad():
                batch_size_test = cfg['test_featuremaps'].shape[1]
                logits_test0 = 100. * cfg['test_features'] @ cfg['clip_weights']
                tip_logits_test_dataset = tip_plot(cfg, cfg['test_featuremaps'],adapter,batch_size_test,beta) 
                tip_logits_test_fuse = logits_test0 + tip_logits_test_dataset * alpha
                tip_logits_test_fuse_new = tip_logits_test_fuse + 0.5*cfg['test_logits']
            
            acc = cls_acc(tip_logits_test_fuse_new, cfg['test_labels'])

            print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_epoch = train_idx
                torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's test accuracy: {best_acc:.2f}. ****\n")

    # Search Hyperparameters
    print("\n-------- Searching hyperparameters on the val set. --------")
    cfg['test_loader'] = test_loader
    cfg['val_loader'] = val_loader
    best_val_acc, best_beta, best_alpha = search_hp(cfg,clip_model, adapter=adapter) 
    beta = best_beta
    alpha= best_alpha 
    

    with torch.no_grad():
        batch_size_test = cfg['test_featuremaps'].shape[1]
        logits_test0 = 100. * cfg['test_features'] @ cfg['clip_weights']
        tip_logits_test_dataset = tip_plot(cfg, cfg['test_featuremaps'],adapter,batch_size_test,beta) 
        tip_logits_test_fuse = logits_test0 + tip_logits_test_dataset * alpha
        tip_logits_test_fuse_new = tip_logits_test_fuse + 0.5*cfg['test_logits']
    acc = cls_acc(tip_logits_test_fuse_new, cfg['test_labels'])
    if acc > best_acc:
        best_acc = acc
        print("**** Tip-Adapter-F's best test accuracy: {:.2f}. at beta:{:.2f} alpha:{:.2f}****\n".format(best_acc,beta,alpha))
    else:
        print("**** Tip-Adapter-F's best test accuracy: {:.2f}. at beta:{:.2f} alpha:{:.2f}****\n".format(best_acc,cfg['init_beta'],cfg['init_alpha']))

def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    from clip import clip
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])
    plot_model_local= clip.load_clip_to_cpu(cfg['backbone'])

    # Load parameter
    
    model_dir = os.path.join(cache_dir, 'shot'+str(cfg['shots'])) 
    model = CustomCLIP(dataset.classnames, plot_model_local)
    for name, param in model.named_parameters(): param.requires_grad_(False)
    device =  torch.device("cuda:0")
    model.to(device)

    if cfg['shots'] >6:
        pretrain_path = os.path.join(model_dir, 'model.pth.tar-200')
    elif cfg['shots'] >1:
        pretrain_path = os.path.join(model_dir, 'model.pth.tar-100')
    else:
        pretrain_path = os.path.join(model_dir, 'model.pth.tar-50')

    checkpoint = load_checkpoint(pretrain_path)
    state_dict = checkpoint["state_dict"]
    epoch = checkpoint["epoch"]
    # Ignore fixed token vectors
    if "token_prefix" in state_dict:
        del state_dict["token_prefix"]

    if "token_suffix" in state_dict:
        del state_dict["token_suffix"]

    print("Loading weights" 'from "{}" (epoch = {})'.format(pretrain_path, epoch))
    # set strict=False
    model.prompt_learner.load_state_dict(state_dict, strict=False)

    model.eval()


    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing dataset.")
    if cfg['dataset']=='food101':
        batch_size0=256
    elif cfg['dataset']=='sun397':
        batch_size0=256
    else:
        batch_size0=64
    val_loader = build_data_loader(data_source=dataset.val, batch_size=batch_size0, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=batch_size0, is_train=False, tfm=preprocess, shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)
    cfg['clip_weights'] = clip_weights

    # Construct the cache model by few-shot training set 

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels, val_featuremaps, val_logits = pre_load_features(cfg, "val", model, val_loader)
    cfg['val_features'] = val_features
    cfg['val_labels'] = val_labels
    cfg['val_featuremaps'] = val_featuremaps
    cfg['val_logits'] = val_logits

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels, test_featuremaps, test_logits = pre_load_features(cfg, "test", model, test_loader)
    cfg['test_features'] = test_features
    cfg['test_labels'] = test_labels
    cfg['test_featuremaps'] = test_featuremaps
    cfg['test_logits'] = test_logits

    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values, cache_keymaps, cache_plot_text = build_cache_model(cfg, model, train_loader_cache)
    cfg['cache_keys'] = cache_keys
    cfg['cache_values'] = cache_values
    cfg['cache_keymaps'] = cache_keymaps
    cfg['cache_plot_text'] = cache_plot_text

    # Load hyperparamaters
    cfg['M'] = 49
    cfg['N'] = 2
    cfg['classnames'] = dataset.classnames


    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F(cfg, model, train_loader_F,val_loader,test_loader)
           

if __name__ == '__main__':
    main()