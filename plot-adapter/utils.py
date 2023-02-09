from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip
import pdb

def Sinkhorn(K, u, v):
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    thresh = 1e-2
    for i in range(100):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break

    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

    return T

def tip_plot(cfg, image_features_map,adapter,batch_size,beta):

    n_cls = len(cfg['classnames'])
    N= cfg['N']
    M = cfg['M']
    k_shot = cfg['shots']
    affinity = adapter(image_features_map)
    affinity = affinity.view(M,batch_size,N,n_cls*k_shot) # M x B x N x (n_cls*k_shot)
    affinity = affinity.permute(0,2,1,3)

    feat_exp = ((-1) * (beta - beta * affinity)).exp()
    feat_exp = F.normalize(feat_exp, dim=-1)
    sim = feat_exp @ (cfg['cache_values'])
    sim = sim.view(M,N,batch_size*n_cls)
    sim = sim.permute(2,0,1)
    wdist = 1.0 - sim

    xx=torch.zeros(batch_size*n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
    yy=torch.zeros(batch_size*n_cls, N, dtype=sim.dtype, device=sim.device).fill_(1. / N)

    with torch.no_grad():
        KK = torch.exp(-wdist / cfg['thre'])
        T = Sinkhorn(KK,xx,yy)
    try:
        torch.isnan(T).any()
    except None:
        print('There is none value in your tensor, please try to adjust #thre and #eps to align data.')

    sim_op = torch.sum(T * sim, dim=(1, 2))
    sim_op = sim_op.contiguous().view(batch_size,n_cls)
    plot_logits = 100. * sim_op

    return plot_logits


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):
    if cfg['load_cache'] == False:  
        cache_keys = []
        cache_values = []
        cache_keymaps = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []
                train_images_features_map = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    _, img_feat, text_feat, img_feat_map = clip_model(images)

                    train_features.append(img_feat)
                    train_images_features_map.append(img_feat_map)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                        if i ==0:
                            cache_plot_text=text_feat

                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
                cache_keymaps.append(torch.cat(train_images_features_map, dim=1).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)

        cache_keymaps = torch.cat(cache_keymaps, dim=0).mean(dim=0)
        cache_keymaps /= cache_keymaps.norm(dim=-1, keepdim=True)
        cache_keymaps = cache_keymaps.permute(2, 0, 1) # d x M x B

        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_keymaps, cfg['cache_dir'] + '/keymaps_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_plot_text, cfg['cache_dir'] + '/plot_text_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
        cache_keymaps = torch.load(cfg['cache_dir'] + '/keymaps_' + str(cfg['shots']) + "shots.pt")
        cache_plot_text = torch.load(cfg['cache_dir'] + '/plot_text_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values, cache_keymaps, cache_plot_text


def pre_load_features(cfg, split, clip_model, loader):
    if cfg['load_pre_feat'] == False:
        features, featuremaps, labels, logits = [], [], [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                logit, img_feat, _, img_feat_map = clip_model(images)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                img_feat_map /= img_feat_map.norm(dim=-1, keepdim=True)

                features.append(img_feat)
                featuremaps.append(img_feat_map)
                labels.append(target)
                logits.append(logit)

        features, labels = torch.cat(features), torch.cat(labels)
        featuremaps = torch.cat(featuremaps,1)
        logits =  torch.cat(logits)
        
        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
        torch.save(featuremaps, cfg['cache_dir'] + "/" + split + "_fm.pt")
        torch.save(logits, cfg['cache_dir'] + "/" + split + "_score.pt")
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
        featuremaps = torch.load(cfg['cache_dir'] + "/" + split + "_fm.pt")
        logits = torch.load(cfg['cache_dir'] + "/" + split + "_score.pt")
    
    return features, labels, featuremaps, logits


def search_hp(cfg, clip_model, adapter=None):

    featuremaps,labels,logits = cfg['val_featuremaps'], cfg['val_labels'], cfg['val_logits']
    loader =  cfg['val_loader'] 
   
    
    beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
    alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

    best_acc = 0
    best_beta, best_alpha = 0, 0

    for beta in beta_list:
        
        if cfg['loop'] ==True:
            tip_logits_test_list_fuse=[]
            tip_logits_test_list = []
            logits_test0_list = []
            for (images, _) in tqdm(loader):
                images = images.cuda()
                with torch.no_grad():
                    _, test_features,_, test_features_map = clip_model(images)
                    test_features /= test_features.norm(dim=-1, keepdim=True)
                    test_features_map /= test_features_map.norm(dim=-1, keepdim=True)
                    test_features_map_batch = test_features_map

                    batch_size_test = test_features.shape[0]
                    logits_test0 = 100. * test_features @ cfg['clip_weights']
                    tip_logits_test= tip_plot(cfg, test_features_map_batch,adapter,batch_size_test,beta)
                    logits_test0_list.append(logits_test0)
                    tip_logits_test_list.append(tip_logits_test)


            for alpha in alpha_list:
                tip_logits_test_fuse_alpha =  torch.cat(logits_test0_list) + torch.cat(tip_logits_test_list) * alpha + 0.5*logits
                acc = cls_acc(tip_logits_test_fuse_alpha, labels)

                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha
        else:
            with torch.no_grad():
                batch_size_test = featuremaps.shape[1]
                logits_test0 = 100. * cfg['val_features'] @ cfg['clip_weights']
                tip_logits_test_dataset = tip_plot(cfg, featuremaps,adapter,batch_size_test,beta) 
            for alpha in alpha_list:
                tip_logits_test_fuse_alpha = logits_test0 + tip_logits_test_dataset * alpha + 0.5*logits
                acc = cls_acc(tip_logits_test_fuse_alpha, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

    print("\nAfter searching, the best accuarcy on the validation set: {:.2f}.\n".format(best_acc))

    return best_acc, best_beta, best_alpha


def search_hp_imagenet(cfg, clip_model, adapter=None):

    labels,logits = cfg['val_labels'], cfg['val_logits']
    loader =  cfg['val_loader'] 
    
    beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
    alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

    best_acc = 0
    best_beta, best_alpha = 0, 0

    for beta in beta_list:
        logits_test0_list=[]
        tip_logits_test_list=[]
        for _, (images, _) in enumerate(tqdm(loader)):
            images = images.cuda()
            with torch.no_grad():
                _, test_features,_, test_features_map = clip_model(images)
                test_features /= test_features.norm(dim=-1, keepdim=True)
                test_features_map /= test_features_map.norm(dim=-1, keepdim=True)
                test_features_map_batch = test_features_map

                batch_size_test = test_features.shape[0]
                logits_test0 = 100. * test_features @ cfg['clip_weights']
                tip_logits_test= tip_plot(cfg, test_features_map_batch,adapter,batch_size_test,beta)
                logits_test0_list.append(logits_test0)
                tip_logits_test_list.append(tip_logits_test)

        for alpha in alpha_list:
            tip_logits_test_fuse_alpha =  torch.cat(logits_test0_list) + torch.cat(tip_logits_test_list) * alpha + 0.5*logits
            acc = cls_acc(tip_logits_test_fuse_alpha, labels)

            if acc > best_acc:
                print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                best_acc = acc
                best_beta = beta
                best_alpha = alpha      

    print("\nAfter searching, the best accuarcy on the testing set: {:.2f}.\n".format(best_acc))

    return best_acc, best_beta, best_alpha