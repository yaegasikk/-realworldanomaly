import torch
from torch import nn
"""
cited from https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch/blob/master/network/anomaly_detector_model.py
"""
class Threelayerfc(nn.Module):
    def __init__(self):
        super(Threelayerfc,self).__init__()
        self.fc1=nn.Linear(1024,512)
        self.relu1=nn.ReLU()
        self.dropout1 = nn.Dropout(0.6)

        self.fc2 = nn.Linear(512,32)
        self.dropout2 = nn.Dropout(0.6)

        self.fc3 = nn.Linear(32,1)
        self.sig = nn.Sigmoid()

        # In the original keras code they use "glorot_normal"
        # As I understand, this is the same as xavier normal in Pytorch
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.sig(out)

        return out


def custom_objective(y_pred, y_true):
    # y_pred (batch_size, 360, 1)
    # y_true (batch_size)
    lambdas = 8e-5

    normal_vids_indices = (y_true == 0).nonzero().flatten()
    anomal_vids_indices = (y_true == 1).nonzero().flatten()

    normal_segments_scores = y_pred[normal_vids_indices]  # (batch/2, 360, 1)
    anomal_segments_scores = y_pred[anomal_vids_indices]  # (batch/2, 360, 1)

    # just for reducing the last dimension
    normal_segments_scores = torch.sum(normal_segments_scores, dim=(-1,))  # (batch/2, 360)
    anomal_segments_scores = torch.sum(anomal_segments_scores, dim=(-1,))  # (batch/2, 360)

    # get the max score for each video
    normal_segments_scores_maxes = normal_segments_scores.max(dim=-1)[0]
    anomal_segments_scores_maxes = anomal_segments_scores.max(dim=-1)[0]

    hinge_loss = 1 - anomal_segments_scores_maxes + normal_segments_scores_maxes
    hinge_loss = torch.max(hinge_loss, torch.zeros_like(hinge_loss))

    """
    Smoothness of anomalous video
    """
    smoothed_scores = anomal_segments_scores[:, 1:] - anomal_segments_scores[:, :-1]
    smoothed_scores_squared = smoothed_scores.pow(2)
    smoothness_loss = smoothed_scores_squared.sum(dim=-1)

    """
    Sparsity of anomalous video
    """
    sparsity_loss = anomal_segments_scores.sum(dim=-1)

    final_loss = (hinge_loss + lambdas*smoothness_loss + lambdas*sparsity_loss).mean()
    return final_loss


class RegularizedLoss(torch.nn.Module):
    def __init__(self, model, original_objective, lambdas=0.001):
        super(RegularizedLoss, self).__init__()
        self.lambdas = lambdas
        self.model = model
        self.objective = original_objective

    def forward(self, y_pred, y_true):
        # loss
        # Our loss is defined with respect to l2 regularization, as used in the original keras code
        fc1_params = torch.cat(tuple([x.view(-1) for x in self.model.fc1.parameters()]))
        fc2_params = torch.cat(tuple([x.view(-1) for x in self.model.fc2.parameters()]))
        fc3_params = torch.cat(tuple([x.view(-1) for x in self.model.fc3.parameters()]))

        l1_regularization = self.lambdas * torch.norm(fc1_params, p=2)
        l2_regularization = self.lambdas * torch.norm(fc2_params, p=2)
        l3_regularization = self.lambdas * torch.norm(fc3_params, p=2)

        return self.objective(y_pred, y_true) + l1_regularization + l2_regularization + l3_regularization