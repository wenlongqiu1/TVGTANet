r""" Chest X-ray few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


# class DatasetLung(Dataset):
#     def __init__(self, datapath, fold, transform, split, shot=1, num_val=600):
#         self.benchmark = 'lung'
#         self.shot = shot
#         self.split = split
#         self.num_val = num_val
#
#         self.base_path = os.path.join(datapath,"Lung Segmentation")
#         self.img_path = os.path.join(self.base_path, 'CXR_png')
#         self.ann_path = os.path.join(self.base_path, 'masks')
#
#         self.categories = ['1']
#
#         self.class_ids = range(0, 1)
#         self.img_metadata_classwise, self.num_images = self.build_img_metadata_classwise()
#
#         self.transform = transform
#
#     def __len__(self):
#         return self.num_images if self.split != 'val' else self.num_val
#
#     def __getitem__(self, idx):
#         query_name, support_names, class_sample = self.sample_episode(idx)
#         query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)
#
#         query_img = self.transform(query_img)
#         query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
#
#         support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
#
#         support_masks_tmp = []
#         for smask in support_masks:
#             smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
#             support_masks_tmp.append(smask)
#         support_masks = torch.stack(support_masks_tmp)
#
#         batch = {'query_img': query_img,
#                  'query_mask': query_mask,
#                  'query_name': query_name,
#                  'support_imgs': support_imgs,
#                  'support_masks': support_masks,
#                  'class_id': torch.tensor(class_sample),
#                  'support_names': support_names,
#
#                  'support_set': [support_imgs, support_masks],
#                  'support_classes': torch.tensor([class_sample])
#                  }
#
#         return batch
#
#     def load_frame(self, query_name, support_names):
#         query_mask = self.read_mask(query_name)
#         support_masks = [self.read_mask(name) for name in support_names]
#
#         query_id = query_name[:-9] + '.png'
#         query_img = Image.open(os.path.join(self.img_path, os.path.basename(query_id))).convert('RGB')
#
#         support_ids = [os.path.basename(name)[:-9] + '.png' for name in support_names]
#         support_names = [os.path.join(self.img_path, sid) for sid in support_ids]
#         support_imgs = [Image.open(name).convert('RGB') for name in support_names]
#
#         return query_img, query_mask, support_imgs, support_masks
#
#     def read_mask(self, img_name):
#         mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
#         mask[mask < 128] = 0
#         mask[mask >= 128] = 1
#         return mask
#
#     def sample_episode(self, idx):
#         class_id = idx % len(self.class_ids)
#         class_sample = self.categories[class_id]
#
#         query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
#         support_names = []
#         while True:  # keep sampling support set if query == support
#             support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
#             if query_name != support_name: support_names.append(support_name)
#             if len(support_names) == self.shot: break
#
#         return query_name, support_names, class_id
#
#     def build_img_metadata(self):
#         img_metadata = []
#         for cat in self.categories:
#             os.path.join(self.base_path, cat)
#             img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.img_path, cat))])
#             for img_path in img_paths:
#                 if os.path.basename(img_path).split('.')[1] == 'png':
#                     img_metadata.append(img_path)
#         return img_metadata
#
#     def build_img_metadata_classwise(self):
#         num_images=0
#         img_metadata_classwise = {}
#         for cat in self.categories:
#             img_metadata_classwise[cat] = []
#
#         for cat in self.categories:
#             img_paths = sorted([path for path in glob.glob('%s/*' % self.ann_path)])
#             for img_path in img_paths:
#                 if os.path.basename(img_path).split('.')[1] == 'png':
#                     img_metadata_classwise[cat] += [img_path]
#                     num_images+=1
#         return img_metadata_classwise, num_images
class DatasetLung(Dataset):
    def __init__(self, datapath, fold, transform, split, shot=1, num_val=600):
        self.benchmark = 'lung'
        self.shot = shot
        self.split = split
        self.num_val = num_val

        self.base_path = os.path.join(datapath, "Lung Segmentation")
        self.img_path = os.path.join(self.base_path, 'CXR_png')
        self.ann_path = os.path.join(self.base_path, 'masks')
        self.pascal_path = os.path.join(self.base_path, 'Pascal')  # 新增Pascal路径

        self.categories = ['1']

        self.class_ids = range(0, 1)
        self.img_metadata_classwise, self.num_images = self.build_img_metadata_classwise()

        self.transform = transform

    def __len__(self):
        return self.num_images if self.split != 'val' else self.num_val

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, query_pascal, support_imgs, support_masks, support_pascals = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_pascal = F.interpolate(query_pascal.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp, support_pascals_tmp = [], []
        for smask, spasc in zip(support_masks, support_pascals):
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            spasc = F.interpolate(spasc.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
            support_pascals_tmp.append(spasc)

        support_masks = torch.stack(support_masks_tmp)
        support_pascals = torch.stack(support_pascals_tmp)

        batch = {
            'query_img': query_img,
            'query_mask': query_mask,
            'query_pascal_label': query_pascal,
            'query_name': query_name,
            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'support_pascal_labels': support_pascals,
            'support_names': support_names,
            'class_id': torch.tensor(class_sample)
        }

        return batch

    def load_frame(self, query_name, support_names):
        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        query_id = query_name[:-9] + '.png'
        query_img = Image.open(os.path.join(self.img_path, os.path.basename(query_id))).convert('RGB')
        query_pascal_path = os.path.join(self.pascal_path, os.path.basename(query_id))
        query_pascal = self.read_mask(query_pascal_path)

        support_ids = [os.path.basename(name)[:-9] + '.png' for name in support_names]
        support_imgs = [Image.open(os.path.join(self.img_path, sid)).convert('RGB') for sid in support_ids]
        support_pascals = [self.read_mask(os.path.join(self.pascal_path, sid)) for sid in support_ids]

        return query_img, query_mask, query_pascal, support_imgs, support_masks, support_pascals

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        class_id = idx % len(self.class_ids)
        class_sample = self.categories[class_id]

        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_id

    def build_img_metadata_classwise(self):
        num_images = 0
        img_metadata_classwise = {cat: [] for cat in self.categories}

        for cat in self.categories:
            img_paths = sorted(glob.glob(f'{self.ann_path}/*.png'))
            for img_path in img_paths:
                img_metadata_classwise[cat].append(img_path)
                num_images += 1

        return img_metadata_classwise, num_images
