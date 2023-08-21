from data import Dataset_Pro
from torch.utils.data import DataLoader
import torch.optim as optim
import time, torch, os
import numpy as np
import torch.nn as nn
from model import ASNet, summaries
import torch.backends.cudnn as cudnn
import pandas as pd
batch_size = 25
epochs = 1000
lr = 0.003
ckpt = 10
model = ASNet().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
criterion = nn.MSELoss(size_average=True).cuda()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.5)
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = True
cudnn.deterministic = True
cudnn.benchmark = False
#   calculate flops and params
# from thop import profile, clever_format
# input1 = torch.randn(1, 8, 64, 64).cuda()
# input2 = torch.randn(1, 1, 256, 256).cuda()
# flops, params = profile(model, inputs= (input1, input2))
# flops = clever_format([flops], '%.3f')
# print('flops: ', flops)
# print('params: %.2f M' % (params / 1000000.0))

satellites = ['WV3','pleiades','IKONOS']
pause_path = ""

def save_checkpoint(model, epoch):
    model_out_path = 'Weights/' + satellites[0] + "/{}.pth".format(epoch)
    state = {'weight': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, model_out_path)

def train(training_data_loader, validate_data_loader, pause_path = ''):
    if os.path.exists(pause_path):
        pausepoint = torch.load(pause_path)
        model.load_state_dict(pausepoint['weight'])
        optimizer.load_state_dict(pausepoint['optimizer'])
        start_epoch=pausepoint['epoch']
    else:
        start_epoch = 0
        log_data = {'epoch':[],'test_loss':[],'valid_loss':[],'time':[]}
        log_data = pd.DataFrame(log_data)
        log_data.to_excel(satellites[0]+'.xlsx',index=False)
    t1 = time.time()
    print('Start training...')
    for epoch in range(start_epoch, epochs, 1):
        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []
        model.train()
        for iteration, batch in enumerate(training_data_loader, 1):
            gt, lrms, pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            optimizer.zero_grad()
            sr = model(lrms, pan)
            loss = criterion(sr, gt)
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        t_loss = np.nanmean(np.array(epoch_train_loss))
        ori_log_file = pd.read_excel(satellites[0]+'.xlsx')
        log_data2 = {'epoch':[epoch],'test_loss':[t_loss],'valid_loss':[None],'time':[None]}
        log_data2 = pd.DataFrame(log_data2)
        save_data = ori_log_file.append(log_data2)
        save_data.to_excel(satellites[0]+'.xlsx', index = False)
        # print('epoch: {}/{},test_loss:{}'.format(epoch,epochs,t_loss))
        # t3 = time.time()
        # print(t3-t1)

        if epoch % ckpt == 0:
            save_checkpoint(model, epoch)
        with torch.no_grad():
            model.eval()
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lrms, pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
                sr = model(lrms, pan)
                loss = criterion(sr, gt)
                epoch_val_loss.append(loss.item())

        if epoch % 10 == 0:
            v_loss = np.nanmean(np.array(epoch_val_loss))
            # print('      validate loss: {:.7f}'.format(v_loss))
            t2 = time.time()
            print('epoch: {}/{}    time cost: {:.4f}s'.format(epoch,epochs,t2 - t1))
            log_data2 = {'epoch':[epoch],'test_loss':[t_loss],'valid_loss':[v_loss],'time':[t2-t1]}
            log_data2 = pd.DataFrame(log_data2)
            save_data = ori_log_file.append(log_data2)
            save_data.to_excel(satellites[0]+'.xlsx', index = False)

if __name__ == "__main__":
    train_set = Dataset_Pro('../../data/H5data/'+satellites[0]+'/train.h5')
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    validate_set = Dataset_Pro('../../data/H5data/'+satellites[0]+'/valid.h5')
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    train(training_data_loader, validate_data_loader, pause_path)

