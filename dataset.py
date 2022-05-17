import numpy as np
import cv2
import os
from torch.utils.data import Dataset


def make_dataset_fromlst(train_dir, label_dir):
    """
    NYUlist format:
    imagepath seglabelpath depthpath HHApath
    """
    NC = []
    ART = []
    PV = []
    label = []
    for filename in os.listdir(train_dir):
        if filename == 'ART':
            temp_path = os.path.join(train_dir, filename)
            for i in os.listdir(temp_path):
                ART.append(os.path.join(temp_path, i))
        elif filename == 'NC':
            temp_path = os.path.join(train_dir, filename)
            for i in os.listdir(temp_path):
                NC.append(os.path.join(temp_path, i))
        elif filename == 'PV':
            temp_path = os.path.join(train_dir, filename)
            for i in os.listdir(temp_path):
                PV.append(os.path.join(temp_path, i))
    if os.path.exists(label_dir):
        for filename in os.listdir(label_dir):
            label.append(os.path.join(label_dir, filename))
        return {'NC': NC, 'ART': ART, 'PV': PV, 'labels': label}
    else:
        return {'NC': NC, 'ART': ART, 'PV': PV}


class FFL(Dataset):
    def __init__(self, data_dir, transform=None, phase_train=True):

        self.phase_train = phase_train
        self.train_dir = data_dir
        self.train_label_dir = data_dir.replace('images', 'labels')
        self.test_dir = data_dir.replace('train', 'test')
        self.test_label_dir = self.test_dir.replace('images', 'labels')
        self.transform = transform

        if self.phase_train:
            result = make_dataset_fromlst(self.train_dir, self.train_label_dir)
            self.NC_dir_train = result['NC']
            self.ART_dir_train = result['ART']
            self.PV_dir_train = result['PV']
            self.label_dir_train = result['labels']
        else:
            result = make_dataset_fromlst(self.test_dir, self.train_label_dir)
            self.NC_dir_test = result['NC']
            self.ART_dir_test = result['ART']
            self.PV_dir_test = result['PV']

    def __len__(self):
        if self.phase_train:
            return len(self.NC_dir_train)
        else:
            return len(self.NC_dir_test)

    def __getitem__(self, index):
        label = ''
        if self.phase_train:
            image_nc = cv2.imread(self.NC_dir_train[index]).astype(np.float32)
            image_art = cv2.imread(self.ART_dir_train[index]).astype(np.float32)
            image_pv = cv2.imread(self.PV_dir_train[index]).astype(np.float32)
            label = self.label_dir_train[index]
        else:
            image_nc = cv2.imread(self.NC_dir_test[index]).astype(np.float32)
            image_art = cv2.imread(self.ART_dir_test[index]).astype(np.float32)
            image_pv = cv2.imread(self.PV_dir_test[index]).astype(np.float32)

        if image_nc.shape[0] > 223 or image_nc.shape[1] > 223:
            image_nc = cv2.resize(image_nc, (224, 224))
        else:
            add_h_nc = int((224 - image_nc.shape[0]) / 2)
            add_w_nc = int((224 - image_nc.shape[1]) / 2)
            image_nc = cv2.copyMakeBorder(image_nc, add_h_nc, 224 - add_h_nc - image_nc.shape[0], add_w_nc,
                                          224 - add_w_nc - image_nc.shape[1],
                                          cv2.BORDER_REFLECT)
        if image_art.shape[0] > 223 or image_art.shape[1] > 223:
            image_art = cv2.resize(image_art, (224, 224))
        else:
            add_h_art = int((224 - image_art.shape[0]) / 2)
            add_w_art = int((224 - image_art.shape[1]) / 2)
            image_art = cv2.copyMakeBorder(image_art, add_h_art, 224 - add_h_art - image_art.shape[0], add_w_art,
                                           224 - add_w_art - image_art.shape[1],
                                           cv2.BORDER_REFLECT)
        if image_pv.shape[0] > 223 or image_pv.shape[1] > 223:
            image_pv = cv2.resize(image_pv, (224, 224))
        else:
            add_h_pv = int((224 - image_pv.shape[0]) / 2)
            add_w_pv = int((224 - image_pv.shape[1]) / 2)
            image_pv = cv2.copyMakeBorder(image_pv, add_h_pv, 224 - add_h_pv - image_pv.shape[0], add_w_pv,
                                          224 - add_w_pv - image_pv.shape[1],
                                          cv2.BORDER_REFLECT)

        if self.phase_train:
            if self.transform:
                for method in self.transform:
                    image_nc, image_art, image_pv, label = method(image_nc, image_art, image_pv, label)
        else:
            image_nc = (image_nc/255).astype(np.float32)
            image_art = (image_art/255).astype(np.float32)
            image_pv = (image_pv/255).astype(np.float32)

        image_nc = image_nc.transpose((2, 0, 1))
        image_art = image_art.transpose((2, 0, 1))
        image_pv = image_pv.transpose((2, 0, 1))

        if self.phase_train:
            sample = {'NC': image_nc, 'ART': image_art, 'PV': image_pv, 'label': label,'index':index}
        else:
            sample = {'NC': image_nc, 'ART': image_art, 'PV': image_pv,'index':index}
        # print(self.NC_dir_train[index])
        return sample

# class RandomScale(object):
#     def __init__(self, scale):
#         self.scale_low = min(scale)
#         self.scale_high = max(scale)
#
#     def __call__(self, sample):
#         image, depth, label = sample['image'], sample['depth'], sample['label']
#
#         target_scale = random.uniform(self.scale_low, self.scale_high)
#         # (H, W, C)
#         target_height = int(round(target_scale * image.shape[0]))
#         target_width = int(round(target_scale * image.shape[1]))
#         # Bi-linear
#         image = skimage.transform.resize(image, (target_height, target_width),
#                                          order=1, mode='reflect', preserve_range=True)
#         # Nearest-neighbor
#         depth = skimage.transform.resize(depth, (target_height, target_width),
#                                          order=0, mode='reflect', preserve_range=True)
#         label = skimage.transform.resize(label, (target_height, target_width),
#                                          order=0, mode='reflect', preserve_range=True)
#
#         return {'image': image, 'depth': depth, 'label': label}
#
#
# class RandomCrop(object):
#     def __init__(self, th, tw):
#         self.th = th
#         self.tw = tw
#
#     def __call__(self, sample):
#         image, depth, label = sample['image'], sample['depth'], sample['label']
#         h = image.shape[0]
#         w = image.shape[1]
#         i = random.randint(0, h - self.th)
#         j = random.randint(0, w - self.tw)
#
#         return {'image': image[i:i + image_h, j:j + image_w, :],
#                 'depth': depth[i:i + image_h, j:j + image_w],
#                 'label': label[i:i + image_h, j:j + image_w]}
#
#
# class RandomFlip(object):
#     def __call__(self, sample):
#         image, depth, label = sample['image'], sample['depth'], sample['label']
#         if random.random() > 0.5:
#             image = np.fliplr(image).copy()
#             depth = np.fliplr(depth).copy()
#             label = np.fliplr(label).copy()
#
#         return {'image': image, 'depth': depth, 'label': label}
#
#
# # Transforms on torch.*Tensor
# class Normalize(object):
#     def __call__(self, sample):
#         image, depth = sample['image'], sample['depth']
#         image = image / 255
#         # image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#         #                                          std=[0.229, 0.112, 0.225])(image)
#         image = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
#                                                  std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(image)
#         depth = torchvision.transforms.Normalize(mean=[2.8424503515351494],
#                                                  std=[0.9932836506164299])(depth)
#         sample['image'] = image
#         sample['depth'] = depth
#
#         return sample
#
#
# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample):
#         image, depth, label = sample['image'], sample['depth'], sample['label']
#
#         # Generate different label scales
#         label2 = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2),
#                                           order=0, mode='reflect', preserve_range=True)
#         label3 = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4),
#                                           order=0, mode='reflect', preserve_range=True)
#         label4 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8),
#                                           order=0, mode='reflect', preserve_range=True)
#         label5 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16),
#                                           order=0, mode='reflect', preserve_range=True)
#
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         image = image.transpose((2, 0, 1))
#         depth = np.expand_dims(depth, 0).astype(np.float)
#         return {'image': torch.from_numpy(image).float(),
#                 'depth': torch.from_numpy(depth).float(),
#                 'label': torch.from_numpy(label).float(),
#                 'label2': torch.from_numpy(label2).float(),
#                 'label3': torch.from_numpy(label3).float(),
#                 'label4': torch.from_numpy(label4).float(),
#                 'label5': torch.from_numpy(label5).float()}
