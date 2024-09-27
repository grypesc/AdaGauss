import torch
import torch.nn.functional as F

from torch.nn import Parameter


def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0.1):
    T = torch.nn.functional.one_hot(T, nb_classes)
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    return T


class ProxyYolo(torch.nn.Module):
    def __init__(self,
                 nb_classes,
                 sz_embedding,
                 device,
                 smoothing=0.1,
                 temperature=1,
                 **kwargs
                 ):
        super().__init__()
        self.proxies = Parameter(torch.randn(nb_classes, sz_embedding, device=device) / 8)
        self.smoothing = smoothing
        self.temperature = temperature


    def forward(self, X, T):
        P = self.proxies
        T = binarize_and_smooth_labels(T, len(P), self.smoothing)
        D = torch.cdist(X, P) ** 2
        loss = torch.sum(-T * F.log_softmax(-D/self.temperature, -1), -1)
        return loss.mean(), None


