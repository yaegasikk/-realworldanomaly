from feature_dataloader import *
from network import *
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

batch_size=60
bag_size = batch_size//2
epochs = 20000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print("Use {}".format(device))
torch.backends.cudnn.benchmark = True

normal_list, anomaly_list = read_trainannotation()
print('Start normal')
"""
#save normal dataset
normal_dataset = FeatureLoader(normal_list,anomaly_list)
pd.to_pickle(normal_dataset,'normal_dataset.pkl')
"""
normal_dataset = pd.read_pickle('normal_dataset.pkl')
#print(normal_dataset[0][0].shape)
print('Start anomaly')
"""
#save normal dataset
anomaly_dataset = FeatureLoader(normal_list,anomaly_list,state='Anomaly')
pd.to_pickle(anomaly_dataset,'anomaly_dataset.pkl')
"""
anomaly_dataset = pd.read_pickle('anomaly_dataset.pkl')

normal_trainloader = torch.utils.data.DataLoader(normal_dataset,batch_size=bag_size,num_workers=4,shuffle=True,drop_last=False)
anomaly_trainloader = torch.utils.data.DataLoader(anomaly_dataset,batch_size=bag_size,num_workers=4,shuffle=True,drop_last=True)

load_epoch = 12000
load_weight = './save_weight/model_{}.pth'.format(load_epoch)
model = Threelayerfc()
model.load_state_dict(torch.load(load_weight))
model = model.to(device)
print('load weight')

criterion = RegularizedLoss(model, custom_objective)
optimizer = torch.optim.Adadelta(model.parameters(),lr=0.01,eps=1e-8)

#print(iter(normal_trainloader).__next__()[0].shape)

add_epochs = epochs+load_epoch
print('Start Train')
for epochs_i in range(add_epochs):
    total_loss = 0
    total_loss_list=[]
    model.train()
    
    for (normal_datas,normal_labels),(anomaly_datas,anomaly_labels) in zip(tqdm(normal_trainloader),anomaly_trainloader):
        #print(anomaly_datas.shape)
        #print(normal_datas.shape)
        if normal_datas.shape[0]==20:
            input_datas = torch.cat([normal_datas,anomaly_datas[:20]])
            input_labels = torch.cat([normal_labels,anomaly_labels[:20]])
        else:
            input_datas = torch.cat([normal_datas,anomaly_datas])
            input_labels = torch.cat([normal_labels,anomaly_labels])
        shuffle_idx = np.random.permutation(np.arange(input_datas.shape[0]))
        input_datas = input_datas[shuffle_idx][:][:].float().to(device)
        input_labels = input_labels[shuffle_idx].long().to(device)


        #print(input_dates[0].is_cuda)
        output = model(input_datas)

        loss = criterion(output,input_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        #print(output.shape)

    total_loss_list.append(total_loss/(len(normal_trainloader)+len(anomaly_trainloader)))
    print("epoch {} , loss {}".format(epochs_i+load_epoch,total_loss_list[-1]))
    if (epochs_i+load_epoch)%2000 == 0:
        torch.save(model.to('cpu').state_dict(),'./save_weight/model_{}.pth'.format(epochs_i+load_epoch))
        model = model.to(device)

torch.save(model.to('cpu').state_dict(),'./save_weight/model_{}.pth'.format(epochs))
    