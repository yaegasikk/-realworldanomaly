import os
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

dataset_dir='./anomaly_features_1fps'

def read_trainannotation(annotation_list='./Train_Annotation.txt'):

    f = open(annotation_list,'r')
    data = f.readlines()
    normalfeature_list =[]
    anomalyfeatire_list =[]
    for data_i in data:
        data_split = data_i.split()
        feature_path = os.path.splitext(data_split[0])[0]+'.npy'
        if os.path.split(feature_path)[0]=='Training_Normal_Videos_Anomaly':
            normalfeature_list.append(feature_path)
        else:
            anomalyfeatire_list.append(feature_path)
    
    return normalfeature_list,anomalyfeatire_list

def feature_transform(feature):
    feature_shape = feature.shape
    if feature_shape[0] >= 360:
        return feature[:360,:]
    else:
        feature_reshape = torch.zeros(360,1024)
        for i,feature_i in enumerate(feature):
            feature_reshape[i] =feature_i[:]
        
        return feature_reshape


class FeatureLoader(data.Dataset):
    def __init__(self,normalfeature_list,anomalyfeatire_list,transforms=feature_transform,dataset_dir='./anomaly_features_1fps',state='Normal'):
        super(FeatureLoader,self).__init__()
        self.dataset_dir=dataset_dir
        self.normalfeature_list=normalfeature_list
        self.anomalyfeatire_list=anomalyfeatire_list
        self.transforms=transforms
        self.state = state
        if self.state == 'Normal':
            self.normal_dataset=(self._get_feature(datalist=self.normalfeature_list))
        else:
            self.anomaly_dataset=self._get_feature(datalist=self.anomalyfeatire_list)
    
    def __len__(self):
        if self.state=='Normal':
            return len(self.normalfeature_list)
        else:
            return len(self.anomalyfeatire_list)

    def _get_feature(self,datalist,dataset_dir=dataset_dir):
        feature_dataset = None
        for datalist_i in tqdm(datalist):
            path = '/'.join([dataset_dir,datalist_i])
            feature = self.transforms(torch.from_numpy(np.load(path)))
            #feature_dataset.append(feature)
            if feature_dataset == None:
                feature_dataset=feature.unsqueeze(0)
            else:
                feature_dataset=torch.cat((feature_dataset,feature.unsqueeze(0)),0)
        print(feature_dataset.shape)
        return feature_dataset

    def __getitem__(self,idx):
        if self.state=='Normal':
            out_label=0
            out_data = self.normal_dataset[idx]
        else:
            out_label=1
            out_data = self.anomaly_dataset[idx]
        
        return out_data,out_label
