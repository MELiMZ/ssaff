import torch.utils.data as data
import torch
import numpy as np
import h5py

class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()     #用于继承父类的属性
        data = h5py.File(file_path)

        gt = data["gt"][...]
        gt = np.array(gt, dtype=np.float32) / 255.     #创建给定元素的数组
        self.gt = torch.from_numpy(gt)      #把数组转换成张量，且二者共享内存

        lrms = data["lrms"][...]
        lrms = np.array(lrms, dtype=np.float32) / 255.
        self.lrms = torch.from_numpy(lrms)

        pan = data['pan'][...]
        pan = np.array(pan, dtype=np.float32) / 255.
        self.pan = torch.from_numpy(pan)

    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), self.lrms[index, :, :, :].float(), self.pan[index, :, :, :].float(),

    def __len__(self):
        return self.gt.shape[0]
