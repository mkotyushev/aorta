import torch
import torch.nn.functional as F


# https://arxiv.org/pdf/2302.03868
def generalized_surface_loss(logits, mask, dtm, weights=None, type='multiclass', reduction='mean'):
    """
    logits: (B, classes, H, W, D), float
    mask: (B, H, W, D), long
    dtm: (B, classes, H, W, D), float
    weights: (classes,), float, sum to 1, optional
    type: str, 'multiclass' or 'multilabel'
    """
    if type == 'multiclass':
        probas = F.softmax(logits, dim=1)
    elif type == 'multilabel':
        probas = F.sigmoid(logits)
    else:
        raise ValueError(f'Unknown type: {type}')
    
    n_classes = probas.shape[1]
    if weights is None:
        weights = torch.ones(n_classes).float().to(probas.device) / probas.size(1)

    mask_onehot = F.one_hot(mask, num_classes=n_classes).float()  # (B, H, W, D, classes)
    mask_onehot = mask_onehot.permute(0, 4, 1, 2, 3)  # (B, classes, H, W, D)

    num = (dtm * (1 - (mask_onehot + probas)) ** 2)  # (B, classes, H, W, D)
    num = num.sum(dim=(2, 3, 4))  # (B, classes)
    num = (weights[None, :] * num).sum(1)  # (B,)

    den = (dtm ** 2).sum(dim=(2, 3, 4))  # (B, classes)
    den = (weights[None, :] * den).sum(1)  # (B,)

    gsl = 1 - num / den  # (B,)
    if reduction == 'mean':
        gsl = gsl.mean()  # scalar
    elif reduction == 'sum':
        gsl = gsl.sum()  # scalar
    else:
        raise ValueError(f'Unknown reduction: {reduction}')

    return gsl
