import os
import random

from .utils import Datum, DatasetBase, listdir_nohidden
from .oxford_pets import OxfordPets


template = ['{} texture.']


class DescribableTextures(DatasetBase):

    dataset_dir = 'dtd'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_DescribableTextures.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)
    
    @staticmethod
    def read_and_split_data(
        image_dir,
        p_trn=0.5,
        p_val=0.2,
        ignored=[],
        new_cnames=None
    ):
        # The data are supposed to be organized into the following structure
        # =============
        # images/
        #     dog/
        #     cat/
        #     horse/
        # =============
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_tst = 1 - p_trn - p_val
        print(f'Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test')

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(
                    impath=im,
                    label=y, # is already 0-based
                    classname=c
                )
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train:n_train+n_val], label, category))
            test.extend(_collate(images[n_train+n_val:], label, category))
        
        return train, val, test
