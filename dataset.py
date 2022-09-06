from tkinter import N, Scale
import utils
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch
import math
from torch.utils.data import DataLoader

class Test_set(Dataset):
    def __init__(self,scale=3,name='Set14'):
        self.foler = [name]
        self.scale = scale
        self._set_()
        
    def __len__(self):
        return len(self.HR)
    
    def __getitem__(self, index):
        lr = self.np2tensor(self.LR[index],scaling=255.)
        hr = self.np2tensor(self.HR[index],scaling=255.)
        return lr,hr
    
    def _set_(self):
        HR,LR = [],[]
        for i in self.foler:
            pth = i + '/' + 'image_SRF_' + str(self.scale)
            for j in os.listdir(pth):
                img = cv2.imread(pth+'/'+j)
                if 'HR' in j:
                    HR.append(img)
                else:
                    LR.append(img)
        self.HR = HR
        self.LR = LR
        
    def np2tensor(self,img,scaling=255):
        img = img.transpose((2,0,1))
        img = torch.from_numpy(img).float()
        return img/scaling
    
class Train_set(Dataset):
    def __init__(self,scale=4,repeat=1):
        self.HR_path = 'DIV2K/DIV2K_train_HR/'
        self.LR_path = 'DIV2K/DIV2K_train_LR_bicubic/x' + str(scale)+'/'
        self.scale = scale
        self.repeat = repeat
        self._set()
        
    def __len__(self):
        return 800*self.repeat
    
    def __getitem__(self, index):
        return self.lr[index],self.hr[index]
    
    def _set(self):
        LR,HR =[],[]
        for i in range(self.repeat):
            lr,hr = self._read()
            lr,hr = self.augment(lr,hr)
            lr,hr = self.get_patch(lr,hr,p=48)
            LR.extend(lr)
            HR.extend(hr)
        self.lr = self.np2tensor(LR,scaling=255.)
        self.hr = self.np2tensor(HR,scaling=255.)
        
    def get_patch(self,lr,hr,p=48):
        if self.rot90:
            iw = np.random.randint(0,self.h_min-p-1)
            ih = np.random.randint(0,self.w_min-p-1)
        else:
            ih = np.random.randint(0,self.h_min-p-1)
            iw = np.random.randint(0,self.w_min-p-1)
        th = ih*self.scale
        tw = iw*self.scale
        tp = p*self.scale
        def _cut(img,ph,pw,p):
            return img[ph:ph+p,pw:pw+p,:]
        lr = [ _cut(a,ih,iw,p) for a in lr]
        hr = [ _cut(a,th,tw,tp) for a in hr]
        return lr,hr
        
    def augment(self,lr,hr):
        hflip = np.random.rand() > 0.5
        vflip = np.random.rand() > 0.5
        self.rot90 = np.random.rand() > 0.5
        def _aug(img):
            if hflip: img = img[:,::-1,:]
            if vflip : img = img[::-1,:,:]
            if self.rot90 : img = img.transpose((1,0,2))
            return img
        lr = [ _aug(a) for a in lr]
        hr = [ _aug(a) for a in hr]
        return lr, hr
    
    def np2tensor(self,_list,scaling=255.):
        _list = np.array(_list)
        _list = np.transpose(_list,(0,3,1,2))
        _list = torch.from_numpy(_list).float()
        _list /= scaling
        return _list
    
    def _read(self):
        LR,HR = [],[]
        h_min,w_min= 1000,1000
        psnr_mean = 0
        for idx in range(800):
            idx = idx % 800
            
            idx = str(idx+1).zfill(4)
            LR_ = cv2.imread(self.LR_path+idx+'x'+str(self.scale)+
                            '.png')
            HR_ = cv2.imread(self.HR_path + idx + '.png')
            
            psnr_mean += PSNR(HR_/255.,cv2.resize(LR_,(0,0),fx=self.scale,fy=self.scale,interpolation=cv2.INTER_CUBIC)/255.)
            h,w = LR_.shape[:2]
            h_min = min(h_min,h)
            w_min = min(w_min,w)
            LR.append(LR_)
            HR.append(HR_)
        print(psnr_mean/800)
        self.h_min = h_min
        self.w_min = w_min
        return LR,HR
    
class Validation_set(Dataset):
    def __init__(self,scale=4):
        self.HR_path = 'DIV2K/DIV2K_train_HR/'
        self.LR_path = 'DIV2K/DIV2K_train_LR_bicubic/x' + str(scale)+'/'
        self.scale = scale
        self.lr,self.hr = self._read()
        
    def __len__(self):
        return 100
    
    def __getitem__(self, index):
        return self.np2tensor(self.lr[index]),self.np2tensor(self.hr[index])
        
    def np2tensor(self,_list,scaling=255.):
        _list = np.array(_list)
        _list = np.transpose(_list,(2,0,1))
        _list = torch.from_numpy(_list).float()
        _list /= scaling
        return _list
    
    def _read(self):
            LR,HR = [],[]
            psnr_mean = 0
            for idx in range(100):
                idx += 800
                idx = str(idx+1).zfill(4)
                LR_ = cv2.imread(self.LR_path+idx+'x'+str(self.scale)+
                                '.png')
                HR_ = cv2.imread(self.HR_path + idx + '.png')

                psnr_mean += PSNR(HR_/255.,cv2.resize(LR_,(0,0),fx=self.scale,fy=self.scale,interpolation=cv2.INTER_CUBIC)/255.)
                LR.append(LR_)
                HR.append(HR_)
            print(psnr_mean/100)
            return LR,HR        

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    
    if(mse == 0):
        return 100
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    
    return psnr 

if __name__ == '__main__':
    test = Validation_set(scale=2)
    print('---------------done---------------')
    #for i in range(10):
    #    lr,hr = test.__getitem__(i)
    #    lr = np.array(lr).transpose(1,2,0)
    #    lr = cv2.resize(lr,(0,0),fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    #    hr = np.array(hr).transpose(1,2,0)
    #    cv2.imwrite(f'test_storage/lr_{i}.png',255*lr)
    #    cv2.imwrite(f'test_storage/hr_{i}.png',255*hr)
        