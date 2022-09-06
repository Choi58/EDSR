from pickle import FALSE
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from dataset import *
from model import *
from learner import *
import time

start_time = time.time()
scale = 2
learning_rate = 0.00000001
batch_size = 16
epochs = 25
n_feats = 256
repeat = 4
train = False
pretraied = True

model = EDSR(scale,n_feats=n_feats).to(device)
if pretraied==True:
    model.load_state_dict(torch.load('model2.pth'))
if train==True:
    training_data = Train_set(scale=scale,repeat=repeat)
    train_dataloader = DataLoader(training_data, batch_size=batch_size,shuffle=True)
validation_data = Validation_set(scale=scale)
valid_dataloader = DataLoader(validation_data, batch_size=1)

for nm in ['Set5','Set14','BSD100']:
    test_data = Test_set(scale=scale,name=nm)
    locals()[f'test_dataloader_{nm}'] = DataLoader(test_data, batch_size=1)

loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             betas=[0.9,0.999],eps=1e-8)
last_psnr = 0
if train==True:
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        #if (count+1) % n_count==0:
        #    count = 0
        #    learning_rate /= 2
        #    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
        #                         betas=[0.9,0.999],eps=1e-8)
        torch.save(model.state_dict(), "model2.pth")
    vv = validation_roop(valid_dataloader,model,False)
    validation_df = pd.DataFrame(vv)
    validation_df.to_excel('valid_df.xlsx')
for nm in ['Set5','Set14','BSD100']:
        test(eval(f'test_dataloader_{nm}'), model, loss_fn,nm)
print(f"Done! time : {time.time()-start_time:7f}")