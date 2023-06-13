import torch
import torch.nn.functional as F

def rce(logits, labels, num_classes=10):
    with torch.no_grad():
        reversed_labels = 1 - F.one_hot(labels, num_classes=num_classes)
        reversed_labels = reversed_labels / (num_classes - 1)
        
    log_pt = F.log_softmax(logits, dim=-1)
    loss = -(log_pt * reversed_labels).sum(-1)
    return loss.mean()
 
