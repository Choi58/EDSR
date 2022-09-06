import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X,y = X.to(device),y.to(device)
        pred = model(X.to(device))
        psnr = _PSNR(y,pred)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 30 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"psnr: {psnr:>7f}  [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model, loss_fn,nm):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    psnr_mean = 0
    with torch.no_grad():
        for idx, U in enumerate(dataloader):
            X, y = U[0].to(device), U[1].to(device)
            pred = model(X)
            psnr_mean += _PSNR(y,pred)
            img_save(pred,idx,nm)
    psnr_mean /= size
    print(f"psnr: {psnr_mean:>7f}")
    
def validation_roop(dataloader, model,train):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    psnr_mean = 0
    psnr_df= {'psnr':[]}
    with torch.no_grad():
        for idx, U in enumerate(dataloader):
            X, y = U[0].to(device), U[1].to(device)
            pred = model(X)
            psnr_mean += _PSNR(y,pred)
            psnr_df['psnr'].append( _PSNR(y,pred) )
            if train==False:
                img_save(pred,idx,'valid')
    psnr_mean /= size
    print(f"psnr: {psnr_mean:>7f}")
    return psnr_df
    
def img_save(pred,idx,nm):
    y = np.array(255*pred.detach().to('cpu'))
    y = np.squeeze(y).transpose((1,2,0))
    cv2.imwrite(f'test_storage/{nm}_{idx}_pred.png',y)
    
def _PSNR(original, compressed):
    original = tensor2np(original)
    compressed = tensor2np(compressed)
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    
    return psnr 

def tensor2np(tensor):
    img = np.array(tensor.detach().to('cpu')).transpose((0,2,3,1))
    return img

