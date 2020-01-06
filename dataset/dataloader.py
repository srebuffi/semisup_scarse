from __future__ import print_function
import math
import torch
import torch.utils.data as data
import torchnet as tnt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.dataloader import default_collate
from .autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy, SubPolicy


# Set the paths of the datasets here.
_CIFAR_DATASET_DIR = './data/cifar'
_SVHN_DATASET_DIR = './data/svhn'


class GenericDataset(data.Dataset):
    def __init__(self, dataset_name, split, autoaugment=False, random_sized_crop=False,
                 num_imgs_per_cat=None):
        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop
        self.autoaugment = autoaugment

        # The num_imgs_per_cats input argument specifies the number
        # of training examples per category that would be used.
        # This input argument was introduced in order to be able
        # to use less annotated examples than what are available
        # in a semi-superivsed experiment. By default all the
        # available training examplers per category are being used.
        self.num_imgs_per_cat = num_imgs_per_cat
        if self.dataset_name == 'cifar10':
            self.n_classes = 10
            self.mean_pix = [0.4914, 0.4822, 0.4465]
            self.std_pix = [0.2023, 0.1994, 0.2010]

            if self.random_sized_crop:
                raise ValueError('The random size crop option is not supported for the CIFAR dataset')

            transform = []
            if (split != 'test'):
                if not self.autoaugment:
                    transform.append(transforms.RandomCrop(32, padding=4))
                    transform.append(transforms.RandomHorizontalFlip())
                else:
                    transform.append(CIFAR10Policy())
            transform.append(lambda x: np.asarray(x))
            self.transform = transforms.Compose(transform)
            self.data = datasets.__dict__[self.dataset_name.upper()](
                _CIFAR_DATASET_DIR, train=self.split == 'train',
                download=True, transform=self.transform
            )
        elif self.dataset_name == 'cifar100':
            self.n_classes = 100
            self.mean_pix = [0.5071, 0.4867, 0.4408]
            self.std_pix = [0.2675, 0.2565, 0.2761]

            if self.random_sized_crop:
                raise ValueError('The random size crop option is not supported for the CIFAR dataset')

            transform = []
            if (split != 'test'):
                transform.append(transforms.RandomCrop(32, padding=4))
                transform.append(transforms.RandomHorizontalFlip())
            transform.append(lambda x: np.asarray(x))
            self.transform = transforms.Compose(transform)
            self.data = datasets.__dict__[self.dataset_name.upper()](
                _CIFAR_DATASET_DIR, train=self.split == 'train',
                download=True, transform=self.transform
            )
        elif self.dataset_name == 'svhn':
            self.n_classes = 10
            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]

            if self.random_sized_crop:
                raise ValueError('The random size crop option is not supported for the SVHN dataset')

            transform = []
            if (split != 'test'):
                transform.append(transforms.RandomCrop(32, padding=4))
            transform.append(lambda x: np.asarray(x))
            self.transform = transforms.Compose(transform)
            self.data = datasets.__dict__[self.dataset_name.upper()](
                _SVHN_DATASET_DIR, split=self.split,
                download=True, transform=self.transform
            )
        else:
            raise ValueError('Not recognized dataset {0}'.format(self.dataset_name))

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def rotate_img(img, rot):
    if rot == 0:  # 0 degrees rotation
        return img
    elif rot == 90:  # 90 degrees rotation
        return np.flipud(np.transpose(img, (1, 0, 2)))
    elif rot == 180:  # 90 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot == 270:  # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1, 0, 2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class DataLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 rotnet=False,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.rotnet = rotnet
        self.num_workers = num_workers

        mean_pix = self.dataset.mean_pix
        std_pix = self.dataset.std_pix
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
        self.inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1, 2, 0).astype(np.uint8),
        ])

    def get_iterator(self, epoch=0):
        # rand_seed = epoch * self.epoch_size
        # random.seed(rand_seed)

        if self.rotnet:
            # if in rotnet mode define a loader function that given the
            # index of an image it returns the 4 rotated copies of the image
            # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
            # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img0, _ = self.dataset[idx]
                rotated_imgs = [
                    self.transform(img0),
                    self.transform(rotate_img(img0, 90).copy()),
                    self.transform(rotate_img(img0, 180).copy()),
                    self.transform(rotate_img(img0, 270).copy())
                ]
                rotation_labels = torch.LongTensor([0, 1, 2, 3])
                return torch.stack(rotated_imgs, dim=0), rotation_labels

            def _collate_fun(batch):
                batch = default_collate(batch)
                assert(len(batch) == 2)
                batch_size, rotations, channels, height, width = batch[0].size()
                batch[0] = batch[0].view([batch_size * rotations, channels, height, width])
                batch[1] = batch[1].view([batch_size * rotations])
                return batch
        else:
            # supervised mode
            # if in supervised mode define a loader function that given the
            # index of an image it returns the image and its categorical label
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, categorical_label = self.dataset[idx]
                img = self.transform(img)
                return img, categorical_label, idx
            _collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.size_epoch()), load=_load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size, collate_fn=_collate_fun, num_workers=self.num_workers, shuffle=self.shuffle
        )
        return data_loader

    def size_epoch(self):
        return len(self.dataset)

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return math.ceil(self.size_epoch() / self.batch_size)


if __name__ == '__main__':
    # Use it for debug purposes
    from matplotlib import pyplot as plt
    dataset = GenericDataset('cifar100', 'test')

    dataloader = DataLoader(dataset, batch_size=1000, rotnet=False, shuffle=True)
    # Get one batch only
    data, label, idxs = next(iter(dataloader()))

    inv_transform = dataloader.inv_transform
    for i in range(data.size(0)):
        plt.subplot(data.size(0) / 4, 4, i + 1)
        inv_data = inv_transform(data[i])
        fig = plt.imshow(inv_data)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.show()
