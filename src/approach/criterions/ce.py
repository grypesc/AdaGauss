import torch
import torch.nn.functional as F



class CE(torch.nn.Module):
    def __init__(self,
                 nb_classes,
                 sz_embedding,
                 device,
                 smoothing=0.0,
                 ):
        super().__init__()
        self.head = torch.nn.Linear(sz_embedding, nb_classes, device=device)
        self.smoothing = smoothing

    def forward(self, features, T):
        logits = self.head(features)
        loss = F.cross_entropy(logits, T, label_smoothing=self.smoothing)

        return loss, logits


