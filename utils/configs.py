import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_list = {'UCMerced': 21, 'AID': 30, 'NWPU': 40, 'RSICB256': 35}

CV_DATA_PATH = {
    'cifar10': '../../dataset',
    'cifar100': '../../dataset'
}

RS_DATA_PATH = {
    'UCMerced': '../../dataset/UCMerced_LandUse',
    'AID': '../../dataset/AID',
    'RSICB256': '../../dataset/RSI-CB256',
    'NWPU': '../../dataset/NWPU-RESISC45'
}
