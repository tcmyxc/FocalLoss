# -*- coding: utf-8 -*-
# 作者：徐文祥(tcmyxc)
# 参考了：
# 1. https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
# 2. https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

import torch
import torch.nn.functional as F


def focal_loss(logits, labels, gamma=2, reduction="mean"):
    r"""
    focal loss for multi classification（简洁版实现）

    `https://arxiv.org/pdf/1708.02002.pdf`

    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    """

    # 这段代码比较简洁，具体可以看作者是怎么定义的，或者看 focal_lossv1 版本的实现
    # 经测试，reduction 加不加结果都一样，但是为了保险，还是加上
    ce_loss = F.cross_entropy(logits, labels, reduction="none")
    log_pt = -ce_loss
    pt = torch.exp(log_pt)
    weights = (1 - pt) ** gamma
    fl = weights * ce_loss

    if reduction == "sum":
        fl = fl.sum()
    elif reduction == "mean":
        fl = fl.mean()
    else:
        raise ValueError(f"reduction '{reduction}' is not valid")
    return fl


def balanced_focal_loss(logits, labels, alpha=0.25, gamma=2, reduction="mean"):
    r"""
    带平衡因子的 focal loss，这里的 alpha 在多分类中应该是个向量，向量中的每个值代表类别的权重。
    但是为了简单起见，我们假设每个类一样，直接传 0.25。
    如果是长尾数据集，则应该自行构造 alpha 向量，同时改写 focal loss 函数。
    """
    return alpha * focal_loss(logits, labels, gamma, reduction)


def focal_lossv1(logits, labels, gamma=2):
    r"""
    focal loss for multi classification（第一版）

    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    """

    # pt = F.softmax(logits, dim=-1)  # 直接调用可能会溢出
    # 一个不会溢出的 trick
    log_pt = F.log_softmax(logits, dim=-1)  # 这里相当于 CE loss
    pt = torch.exp(log_pt)  # 通过 softmax 函数后打的分
    labels = labels.view(-1, 1)  # 多加一个维度，为使用 gather 函数做准备
    pt = pt.gather(1, labels)  # 挑选出真实值对应的 softmax 打分，也可以使用独热编码实现
    ce_loss = -torch.log(pt)
    weights = (1 - pt) ** gamma
    fl = weights * ce_loss
    fl = fl.mean()
    return fl


if __name__ == "__main__":
    logits = torch.tensor([[0.3, 0.6, 0.9, 1], [0.6, 0.4, 0.9, 0.5]])
    labels = torch.tensor([1, 3])
    print(focal_loss(logits, labels))
    print(focal_loss(logits, labels, reduction="sum"))
    print(focal_lossv1(logits, labels))
    print(balanced_focal_loss(logits, labels))
