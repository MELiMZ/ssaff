import os, time
import h5py
import torch
import numpy as np
import scipy.io as sio
from model import ASNet

satellites = ['IKONOS','pleiades','WV3']
ckpt_path = './Weights/'+satellites[0]+'/1000.pth'
file_path = '../../data/H5data/'+satellites[0]



model = ASNet().cuda().eval()
pausepoint = torch.load(ckpt_path)
model.load_state_dict(pausepoint['weight'])  

#RR

def load_set_RR(file_path):
    data = h5py.File(file_path)
    lrms = data['ms'][...]
    lrms = np.array(lrms, dtype=np.float32) / 255.
    lrms = torch.from_numpy(lrms)#.permute(0, 3, 1, 2)

    pan = data['pan'][...]
    pan = np.array(pan, dtype=np.float32) / 255.
    pan = torch.from_numpy(pan)

    gt = data['gt'][...]
    gt = np.array(gt, dtype=np.float32) / 255.
    gt = torch.from_numpy(gt)
    return lrms, pan, gt

def testRR(file_path):
    lrms, pan, gt = load_set_RR(file_path)
    RR_time = 0.0
    with torch.no_grad():
        for i in range(lrms.shape[0]):
            t1 = time.time()
            x1, x2, x3 = lrms[i, :, :, :], pan[i, :, :, :], gt[i, :, :, :]

            x1 = x1.cuda().unsqueeze(dim=0).float()
            x2 = x2.cuda().unsqueeze(dim=0).float()
            x3 = x3.cuda().float()
            sr = model(x1, x2)
            sr = torch.clamp(sr, 0, 1)
            x1 = torch.clamp(x1, 0, 1)
            x2 = torch.clamp(x2, 0, 1)
            x3 = torch.clamp(x3, 0, 1)
            sr = torch.squeeze(sr).permute(1, 2, 0).cpu().detach().numpy()
            x2 = torch.squeeze(x2).cpu().detach().numpy()
            x1 = torch.squeeze(x1).permute(1, 2, 0).cpu().detach().numpy()
            x3 = torch.squeeze(x3).permute(1, 2, 0).cpu().detach().numpy()
            RR_time += time.time() - t1

            sr_save_name = os.path.join('results/'+satellites[0]+'/RR/fushion', '{}.mat'.format(i))
            gt_save_name = os.path.join('results/'+satellites[0]+'/RR/gt', '{}.mat'.format(i))
            pan_save_name = os.path.join('results/'+satellites[0]+'/RR/pan', '{}.mat'.format(i))
            ms_save_name = os.path.join('results/'+satellites[0]+'/RR/lrms', '{}.mat'.format(i))
            
            sr = sr * 256.
            x1 = x1 * 256.
            x2 = x2 * 256.
            x3 = x3 * 256.

            sio.savemat(sr_save_name, {'sr': sr})
            sio.savemat(gt_save_name, {'x3': x3})
            sio.savemat(pan_save_name, {'x2': x2})
            sio.savemat(ms_save_name, {'x1': x1})
    RR_time = RR_time / lrms.shape[0]
    print(RR_time)

def load_set_FR(file_path):
    data = h5py.File(file_path)
    lrms = data['ms'][...]
    lrms = np.array(lrms, dtype=np.float32) / 255.
    lrms = torch.from_numpy(lrms)#.permute(0, 3, 1, 2)

    pan = data['pan'][...]
    pan = np.array(pan, dtype=np.float32) / 255.
    pan = torch.from_numpy(pan)

    return lrms, pan

def testFR(file_path):
    lrms, pan= load_set_FR(file_path)
    FR_time = 0
    with torch.no_grad():
        for i in range(lrms.shape[0]):
            t1 = time.time()
            x1, x2 = lrms[i, :, :, :], pan[i, :, :, :]

            x1 = x1.cuda().unsqueeze(dim=0).float()
            x2 = x2.cuda().unsqueeze(dim=0).float()
            sr = model(x1, x2)
            sr = torch.clamp(sr, 0, 1)
            x1 = torch.clamp(x1, 0, 1)
            x2 = torch.clamp(x2, 0, 1)
            sr = torch.squeeze(sr).permute(1, 2, 0).cpu().detach().numpy()
            x2 = torch.squeeze(x2).cpu().detach().numpy()
            x1 = torch.squeeze(x1).permute(1, 2, 0).cpu().detach().numpy()
            FR_time += time.time() - t1

            sr_save_name = os.path.join('results/'+satellites[0]+'/FR/fushion', '{}.mat'.format(i))
            pan_save_name = os.path.join('results/'+satellites[0]+'/FR/pan', '{}.mat'.format(i))
            ms_save_name = os.path.join('results/'+satellites[0]+'/FR/lrms', '{}.mat'.format(i))

            sr = sr * 256.
            x1 = x1 * 256.
            x2 = x2 * 256.

            sio.savemat(sr_save_name, {'sr': sr})
            sio.savemat(pan_save_name, {'x2': x2})
            sio.savemat(ms_save_name, {'x1': x1})
    print(FR_time / lrms.shape[0])

if __name__ == '__main__':
    file_path_RR = file_path + '/test.h5'
    file_path_FR = file_path + '/withoutRef.h5'
    print("RR begin")
    testRR(file_path_RR)
    print("FR begin")
    testFR(file_path_FR)
    print("done")
