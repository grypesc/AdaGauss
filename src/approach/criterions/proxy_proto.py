import torch
from torch.nn import Parameter
import torch.nn.functional as F


def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0.1):
    T = torch.nn.functional.one_hot(T, nb_classes)
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    return T


class ProxyProto(torch.nn.Module):
    def __init__(self,
                 nb_classes,
                 sz_embedding,
                 device,
                 smoothing=0.1,
                 temperature=1,
                 scaling_p=3
                 ):
        super().__init__()
        # initialize proxies s.t. norm of each proxy ~1 through div by 8
        # i.e. proxies.norm(2, dim=1)) should be close to [1,1,...,1]
        # TODO: use norm instead of div 8, because of embedding size
        self.proxies = Parameter(torch.randn(nb_classes, sz_embedding, device=device) / 8)
        self.smoothing = smoothing
        self.temperature = temperature
        self.scaling_p = scaling_p

    def forward(self, X, T, old_proxies, temperature=1.0):
        P = F.normalize(self.proxies, p=2, dim=-1) * self.scaling_p
        if old_proxies is not None:
            O = F.normalize(old_proxies, p=2, dim=-1) * self.scaling_p
            P = torch.cat((O, P), dim=0)
        X = F.normalize(X, p=2, dim=-1)
        D = torch.cdist(X, P) ** 2
        T = binarize_and_smooth_labels(T, len(P), self.smoothing)
        loss = torch.sum(-T * F.log_softmax(-D/self.temperature, -1), -1)
        return loss.mean(), None


