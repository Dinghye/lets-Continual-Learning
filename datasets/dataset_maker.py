from torchvision.transforms import transforms
from datasets import *
from utils.configs import RS_DATA_PATH, CV_DATA_PATH


def dataset_maker(dataset, img_size):
    if dataset == 'RSICB256':
        train_transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # todo :not clc yet
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        test_transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = iRSICB256(RS_DATA_PATH[dataset], train=True,
                                  train_transform=train_transform)
        test_dataset = iRSICB256(RS_DATA_PATH[dataset], train=False, test_transform=test_transform)

    if dataset == 'UCMerced':
        train_transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        test_transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = iUCMerced(RS_DATA_PATH[dataset], train=True,
                                  train_transform=train_transform)
        test_dataset = iUCMerced(RS_DATA_PATH[dataset], train=False,
                                 test_transform=test_transform)
    if dataset == 'AID':
        train_transform = transforms.Compose([
            # transforms.RandomCrop(img_size, padding=4),
            transforms.Resize([img_size, img_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # TODO: not clc yet
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            # transforms.RandomCrop(img_size, padding=4),
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = iAID(RS_DATA_PATH[dataset], train=True,
                             train_transform=train_transform)
        test_dataset = iAID(RS_DATA_PATH[dataset], train=False, test_transform=test_transform)
    if dataset == 'NWPU':
        train_transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # TODO: not clc yet
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = iNWPU(RS_DATA_PATH[dataset], train=True,
                              train_transform=train_transform)
        test_dataset = iNWPU(RS_DATA_PATH[dataset], train=False, test_transform=test_transform)

    if dataset == 'cifar100':
        train_transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.24705882352941178),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        test_transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        train_dataset = iCIFAR100(CV_DATA_PATH[dataset], transform=train_transform, download=True)
        test_dataset = iCIFAR100(CV_DATA_PATH[dataset], test_transform=test_transform, train=False, download=True)

    if dataset == 'cifar10':
        train_transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.RandomCrop((32, 32), padding=4),
            # transforms.Resize(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.24705882352941178),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        test_transform = transforms.Compose([
            # transforms.Resize(224),

            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_dataset = iCIFAR10(CV_DATA_PATH[dataset], train=True, transform=train_transform,
                                 download=True)
        test_dataset = iCIFAR10(CV_DATA_PATH[dataset], train=False, transform=test_transform,
                                download=True)

    return train_dataset, test_dataset
