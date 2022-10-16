import contextlib
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CE_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        y_true = torch.clamp(y_true, 1e-7, 1 - 1e-7)
        loss = -y_true * torch.log(F.softmax(y_pred) + 1e-7)
        loss_prop = torch.sum(loss, dim=-1).mean()
        return loss_prop


class Unbiased(nn.Module):
    def __init__(self, theta, Class_Number, Bag_Number):
        super().__init__()
        self.theta = theta
        self.Class_Number = Class_Number
        self.Bag_Number = Bag_Number

    def forward(self, y_true, y_pred):
        theta_T = self.theta.pinverse()
        device = y_pred.device
        y_true_ = torch.tensor(torch.unsqueeze(y_true, 1), dtype=torch.long)
        one_hot = torch.zeros(len(y_true_), self.Bag_Number).scatter_(1, y_true_.cpu(), 1)
        one_hot_ = one_hot.permute(1, 0)

        risk_ = torch.ones(self.Bag_Number).to(device)
        for i in range(self.Bag_Number):
            y_pred_i_ = y_pred[one_hot_[i] == 1]
            if len(y_pred_i_) == 0:
                continue
            bag_risk = torch.ones(self.Class_Number).to(device)
            for j in range(self.Class_Number):
                loss = nn.CrossEntropyLoss()
                label_j = torch.tensor(torch.ones(len(y_pred_i_)) * j, dtype=torch.long).to(device)
                output = loss(y_pred_i_, label_j)
                bag_risk[j] = (output * theta_T[j][i]).to(device)
            risk_[i] = torch.sum(bag_risk)
        risk = torch.sum(risk_)
        return risk


class U_PRR(nn.Module):
    def __init__(self, theta, Class_Number, Bag_Number, Gradent_Ascent, Combination):
        super().__init__()
        self.theta = theta
        self.Class_Number = Class_Number
        self.Bag_Number = Bag_Number
        self.Gradent_Ascent = Gradent_Ascent
        self.Combination = Combination

    def forward(self, y_true, y_pred):
        theta_T = self.theta.pinverse()
        device = y_pred.device
        y_true_ = torch.tensor(torch.unsqueeze(y_true, 1), dtype=torch.long)
        one_hot = torch.zeros(len(y_true_), self.Bag_Number).scatter_(1, y_true_.cpu(), 1)
        one_hot_ = one_hot.permute(1, 0)

        risk_ = torch.ones(self.Bag_Number).to(device)
        for i in range(self.Bag_Number):
            y_pred_i_ = y_pred[one_hot_[i] == 1]
            if len(y_pred_i_) == 0:
                continue
            bag_risk = torch.ones(self.Class_Number).to(device)
            float = torch.ones(self.Class_Number).to(device)
            for j in range(self.Class_Number):
                loss = nn.CrossEntropyLoss()
                label_j = torch.tensor(torch.ones(len(y_pred_i_)) * j, dtype=torch.long).to(device)
                output = loss(y_pred_i_, label_j)

                bag_risk[j] = (output * theta_T[j][i]).to(device)

                y_pred_i_list = torch.max(y_pred_i_, 1)[1].double()
                y_diff = y_pred_i_list - label_j
                partial_risk_01 = torch.tensor((len(y_pred_i_) - len(y_diff[y_diff == 0])) / len(y_pred_i_),
                                               dtype=torch.float64).to(device)
                float[j] = torch.where(
                    partial_risk_01.double() < (1 - self.theta[i][j].to(device)),
                    - self.Gradent_Ascent * (output - (1 - self.theta[i][j]).to(device)),
                    (output - (1 - self.theta[i][j]).to(device))
                ) * torch.abs(theta_T[j][i]).to(device)  # 系数
            risk_[i] = (1 - self.Combination) * torch.sum(bag_risk) + self.Combination * torch.sum(float)
        risk = torch.sum(risk_)
        return risk


class U_correct(nn.Module):
    def __init__(self, theta, Class_Number, Bag_Number, Gradent_Ascent):
        super().__init__()
        self.theta = theta
        self.Class_Number = Class_Number
        self.Bag_Number = Bag_Number
        self.Gradent_Ascent = Gradent_Ascent

    def forward(self, y_true, y_pred):
        theta_T = self.theta.pinverse()
        y_true_ = torch.tensor(torch.unsqueeze(y_true, 1), dtype=torch.long)
        one_hot = torch.zeros(len(y_true_), self.Bag_Number).scatter_(1, y_true_.cpu(), 1)
        one_hot_ = one_hot.permute(1, 0)

        risk_ = torch.ones(self.Class_Number)
        for j in range(self.Class_Number):
            class_risk = torch.ones(self.Bag_Number)
            for i in range(self.Bag_Number):
                y_pred_i_ = y_pred[one_hot_[i] == 1]
                if len(y_pred_i_) == 0:
                    continue
                label_j = torch.tensor(torch.ones(len(y_pred_i_)) * j, dtype=torch.long)

                loss = nn.CrossEntropyLoss()
                output = loss(y_pred_i_, label_j)

                class_risk[i] = (output * theta_T[j][i])
            risk_[j] = torch.where(
                torch.sum(class_risk) < 0,
                - torch.sum(class_risk) * self.Gradent_Ascent,
                torch.sum(class_risk)
            )
        risk = torch.sum(risk_)
        return risk


class U_flood(nn.Module):
    def __init__(self, theta, Class_Number, Bag_Number, flood_level):
        super().__init__()
        self.theta = theta
        self.Class_Number = Class_Number
        self.Bag_Number = Bag_Number
        self.flood_level = flood_level

    def forward(self, y_true, y_pred):
        theta_T = self.theta.pinverse()
        device = y_pred.device
        y_true_ = torch.tensor(torch.unsqueeze(y_true, 1), dtype=torch.long)
        one_hot = torch.zeros(len(y_true_), self.Bag_Number).scatter_(1, y_true_.cpu(), 1)
        one_hot_ = one_hot.permute(1, 0)

        risk_ = torch.ones(self.Bag_Number).to(device)
        for i in range(self.Bag_Number):
            y_pred_i_ = y_pred[one_hot_[i] == 1]
            if len(y_pred_i_) == 0:
                continue
            bag_risk = torch.ones(self.Class_Number).to(device)
            for j in range(self.Class_Number):
                loss = nn.CrossEntropyLoss()
                label_j = torch.tensor(torch.ones(len(y_pred_i_)) * j, dtype=torch.long).to(device)
                output = loss(y_pred_i_, label_j)
                bag_risk[j] = (output * theta_T[j][i]).to(device)
            risk_[i] = torch.sum(bag_risk)
        risk = torch.abs(torch.sum(risk_) - self.flood_level) + self.flood_level
        return risk


class Prop(nn.Module):
    def __init__(self, theta, Class_Number, Bag_number):
        super().__init__()
        self.theta = theta
        self.Class_Number = Class_Number
        self.Bag_Number = Bag_number

    def forward(self, y_true, y_pred):
        y_true_ = torch.tensor(torch.unsqueeze(y_true, 1), dtype=torch.long)
        one_hot = torch.zeros(len(y_true_), self.Bag_Number).scatter_(1, y_true_, 1)
        one_hot_ = one_hot.permute(1, 0)

        P_loss = torch.ones(self.Bag_Number)
        for i in range(self.Bag_Number):
            y_pred_i_ = y_pred[one_hot_[i] == 1]
            y_pred_i = torch.sum(y_pred_i_, 0) / len(y_pred_i_)
            true_prop = torch.tensor(self.theta[i], dtype=torch.float32)
            loss_temp = CE_loss()
            P_loss[i] = loss_temp(true_prop, y_pred_i)
        ProportionLoss = torch.sum(P_loss)
        return ProportionLoss


# -------------------------VAT--------------------------------

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        # d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = torch.randn_like(x)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds

