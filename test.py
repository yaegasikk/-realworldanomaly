from feature_dataloader import *
from network import *
import torch
import numpy as np

from sklearn.metrics import roc_curve,roc_auc_score,auc
import matplotlib.pyplot as plt
import pandas as pd

def main():
    model_weight='./save_weight/model_12000.pth'
    model = Threelayerfc()
    model.load_state_dict(torch.load(model_weight))
    print('load weight')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Use {}'.format(device))
    
    """
    dataset = TestFeatureLoader()
    pd.to_pickle(dataset,'test_dataset.pkl')
    """
    dataset = pd.read_pickle('test_dataset.pkl')

    test_dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,num_workers=4,shuffle=False)
    model.to(device)
    model.eval()

    pred_score = np.array([])
    gt_score = np.array([])

    with torch.no_grad():
        for feature_i,label_i in tqdm(test_dataloader):
            feature_i=feature_i.float().to(device)

            output = model(feature_i)
            output = output.to('cpu').squeeze(0).numpy()
            #print(output.shape)
            pred_score =np.append(pred_score,output)
            gt_score = np.append(gt_score,label_i.squeeze(0).numpy())
    
    print(gt_score.shape)

    np.save('./pred_score.npy',pred_score)
    np.save('./gt_score',gt_score)
    
    fpr, tpr, thresholds = roc_curve(gt_score, pred_score)
    plt.figure()
    lw = 2
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    
    plt.savefig('./sklearn_roc_curve.png')
    auc_score = auc(fpr,tpr)
    print('auc: {}'.format(auc_score))


if __name__=="__main__":
    main()

