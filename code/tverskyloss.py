import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)

        true_positive = torch.sum(y_pred * y_true)
        false_positive = torch.sum(y_pred) - true_positive
        false_negative = torch.sum(y_true) - true_positive


        tversky_index = (true_positive+self.smooth)/(
            (true_positive)+
            (self.alpha*false_positive)+
            (self.beta*false_negative)+
            (self.smooth))

        return 1 - tversky_index
    

# class TverskyLoss(nn.Module):
#     def __init__(self, n_classes, alpha=0.5, beta=0.5, gamma=1, smooth=1e-5):
#         super(TverskyLoss, self).__init__()
#         self.n_classes = n_classes
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.smooth = smooth

#     def _one_hot_encoder(self, input_tensor):
#         tensor_list = []
#         print(input_tensor.size())
#         for i in range(self.n_classes):
#             temp_tensor = input_tensor == i
#             tensor_list.append(temp_tensor.unsqueeze(0))
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor

#     def _tversky_loss(self, score, target):

#         target = target.float()
#         true_positive = torch.sum(score*target)
#         false_positive = torch.sum(score) - true_positive
#         false_negative = torch.sum(target) - true_positive

#         tversky_index = (true_positive+self.smooth)/(
#                     (true_positive)+
#                     (self.alpha*false_positive)+
#                     (self.beta*false_negative)+
#                     (self.smooth))
#         return 1 - tversky_index

#     def forward(self, inputs, target, weight=None, softmax=False):
#         print(inputs.size(),target.size())
#         if softmax:
#             inputs = torch.softmax(inputs, dim=1)
#         else:
#             inputs = torch.sigmoid(inputs)

#         print('size before one hot :',target.size())
#         target = self._one_hot_encoder(target)

#         # #flatten label and prediction tensors
#         # inputs = inputs.view(-1)
#         # target = target.view(-1)

#         if weight is None:
#             weight = [1] * self.n_classes

#         assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
#         class_wise_tversky = []
#         loss = 0.0
#         for i in range(0, self.n_classes):
#             tversky = self._tversky_loss(inputs[:, i], target[:, i])
#             class_wise_tversky.append(1.0 - tversky.item())
#             loss += tversky * weight[i]
#         return loss / self.n_classes


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        else:
            inputs = torch.sigmoid(inputs)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes