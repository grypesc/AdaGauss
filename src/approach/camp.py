import copy
import random
import torch
import numpy as np

from argparse import ArgumentParser
from itertools import compress
from torch import nn
from torch.utils.data import Dataset
from torchmetrics import Accuracy

from .mvgb import ClassMemoryDataset, ClassDirectoryDataset
from .models.resnet32 import resnet8, resnet14, resnet20, resnet32
from .models.resnet18 import resnet18
from .incremental_learning import Inc_Learning_Appr
from .criterions.proxy_nca import ProxyNCA
from .criterions.ce import CE


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=1,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, patience=5, fix_bn=False, eval_on_train=False,
                 logger=None, N=10, freeze_after=999, K=3, S=64, distiller="linear", criterion="proxy-nca", alpha=0.5, smoothing=0., sval_fraction=0.95, adapt=False, activation_function="relu", nnet="resnet32"):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)

        self.N = N
        self.K = K
        self.S = S
        self.adapt = adapt
        self.alpha = alpha
        self.smoothing = smoothing
        self.patience = patience
        self.old_model = None
        self.model = {"resnet8": resnet8(num_features=S, activation_function=activation_function),
                      "resnet14": resnet14(num_features=S, activation_function=activation_function),
                      "resnet18": resnet18(num_features=S),
                      "resnet20": resnet20(num_features=S, activation_function=activation_function),
                      "resnet32": resnet32(num_features=S, activation_function=activation_function)}[nnet]
        self.model.fc = nn.Identity()
        self.model.to(device, non_blocking=True)
        self.train_data_loaders, self.val_data_loaders = [], []
        self.prototypes = torch.empty((0, self.S), device=self.device)
        self.task_offset = [0]
        self.classes_in_tasks = []
        self.criterion = {"proxy-nca": ProxyNCA,
                          "ce" : CE}[criterion]
        self.freeze_after = freeze_after
        self.sval_fraction = sval_fraction
        self.svals_explained_by = []
        self.distiller_type = distiller



    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--N',
                            help='Number of learners',
                            type=float,
                            default=100)
        parser.add_argument('--K',
                            help='number of learners sampled for task',
                            type=int,
                            default=3)
        parser.add_argument('--S',
                            help='latent space size',
                            type=int,
                            default=64)
        parser.add_argument('--freeze-after',
                            help='latent space size',
                            type=int,
                            default=999)
        parser.add_argument('--alpha',
                            help='relative weight of kd loss',
                            type=float,
                            default=0.8)
        parser.add_argument('--sval-fraction',
                            help='Fraction of eigenvalues sum that is explained',
                            type=float,
                            default=0.95)
        parser.add_argument('--adapt',
                            help='Adapt prototypes',
                            action='store_true',
                            default=True)
        parser.add_argument('--activation-function',
                            help='Activation functions in resnet',
                            type=str,
                            choices=["identity", "relu", "lrelu"],
                            default="relu")
        parser.add_argument('--distiller',
                            help='Distiller',
                            type=str,
                            choices=["linear", "mlp"],
                            default="mlp")
        parser.add_argument('--criterion',
                            help='Loss function',
                            type=str,
                            choices=["ce", "proxy-nca", "abc"],
                            default="proxy-nca")
        parser.add_argument('--smoothing',
                            help='label smoothing',
                            type=float,
                            default=0.0)
        parser.add_argument('--nnet',
                            type=str,
                            choices=["resnet8", "resnet14", "resnet20", "resnet32", "resnet18"],
                            default="resnet32")
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        num_classes_in_t = len(np.unique(trn_loader.dataset.labels))
        self.classes_in_tasks.append(num_classes_in_t)
        self.train_data_loaders.extend([trn_loader])
        self.val_data_loaders.extend([val_loader])
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        self.task_offset.append(num_classes_in_t + self.task_offset[-1])
        print("### Training backbone ###")
        if t < self.freeze_after:
            self.train_backbone(t, trn_loader, val_loader, num_classes_in_t)
            if t > 0 and self.adapt:
                print("### Adapting prototypes ###")
                self.adapt_prototypes(t, trn_loader, val_loader)
        # torch.save(self.model.state_dict(), f"{self.logger.exp_path}/model_{t}.pth")
        print("### Creating new prototypes ###")
        self.create_prototypes(t, trn_loader, val_loader, num_classes_in_t)
        self.check_singular_values(t, val_loader)
        self.print_singular_values()

        print("Proto norm statistics:")
        norms = torch.norm(self.prototypes, dim=1)
        print(f"Mean: {norms.mean():.2f}, median: {norms.median():.2f}")
        print(f"Range: [{norms.min():.2f}; {norms.max():.2f}]")

    def train_backbone(self, t, trn_loader, val_loader, num_classes_in_t):
        print(f'The model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters')
        print(f'The expert has {sum(p.numel() for p in self.model.parameters() if not p.requires_grad):,} shared parameters\n')
        distiller = nn.Linear(self.S, self.S)
        if self.distiller_type == "mlp":
            distiller = nn.Sequential(nn.Linear(self.S, 2 * self.S),
                                      nn.GELU(),
                                      nn.Linear(2 * self.S, self.S)
                                      )

        distiller.to(self.device, non_blocking=True)
        criterion = self.criterion(num_classes_in_t, self.S, self.device)
        parameters = list(self.model.parameters()) + list(criterion.parameters()) + list(distiller.parameters())
        optimizer, lr_scheduler = self.get_optimizer(parameters, self.wd * (t == 0))

        for epoch in range(self.nepochs):
            train_loss, train_kd_loss, valid_loss, valid_kd_loss = [], [], [], []
            train_hits, val_hits = 0, 0
            self.model.train()
            criterion.train()
            distiller.train()
            for images, targets in trn_loader:
                targets -= self.task_offset[t]
                bsz = images.shape[0]
                images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                features = self.model(images)
                if epoch < 10 and t > 0:
                    features = features.detach()
                loss, logits = criterion(features, targets)
                with torch.no_grad():
                    old_features = self.old_model(images) if t > 0 else None

                total_loss, kd_loss = self.distill_knowledge(loss, features, distiller, old_features)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, self.clipgrad)
                optimizer.step()
                if logits is not None:
                    train_hits += float(torch.sum((torch.argmax(logits, dim=1) == targets)))
                train_loss.append(float(bsz * loss))
                train_kd_loss.append(float(bsz * kd_loss))
            lr_scheduler.step()

            self.model.eval()
            criterion.eval()
            distiller.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    targets -= self.task_offset[t]
                    bsz = images.shape[0]
                    images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                    features = self.model(images)
                    loss, logits = criterion(features, targets)
                    old_features = self.old_model(images) if t>0 else None

                    _, kd_loss = self.distill_knowledge(loss, features, distiller, old_features)
                    if logits is not None:
                        val_hits += float(torch.sum((torch.argmax(logits, dim=1) == targets)))
                    valid_loss.append(float(bsz * loss))
                    valid_kd_loss.append(float(bsz * kd_loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            train_kd_loss = sum(train_kd_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            valid_kd_loss = sum(valid_kd_loss) / len(val_loader.dataset)

            train_acc = train_hits / len(trn_loader.dataset)
            val_acc = val_hits / len(val_loader.dataset)

            print(f"Epoch: {epoch} Train: {train_loss:.2f} KD: {train_kd_loss:.3f} Acc: {100 * train_acc:.2f} "
                  f"Val: {valid_loss:.2f} KD: {valid_kd_loss:.3f} Acc: {100 * val_acc:.2f}")

    @torch.no_grad()
    def create_prototypes(self, t, trn_loader, val_loader, num_classes_in_t):
        """ Create distributions for task t"""
        self.model.eval()
        transforms = val_loader.dataset.transform
        model = self.model
        new_protos = torch.zeros((num_classes_in_t, self.S), device=self.device)
        for c in range(num_classes_in_t):
            train_indices = torch.tensor(trn_loader.dataset.labels) == c + self.task_offset[t]
            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(trn_loader.dataset.images, train_indices))
                ds = ClassDirectoryDataset(train_images, transforms)
            else:
                ds = trn_loader.dataset.images[train_indices]
                ds = ClassMemoryDataset(ds, transforms)
            loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)
            from_ = 0
            class_features = torch.full((2 * len(ds), self.S), fill_value=-999999999.0, device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                features = model(images)
                class_features[from_: from_+bsz] = features
                features = model(torch.flip(images, dims=(3,)))
                class_features[from_+bsz: from_+2*bsz] = features
                from_ += 2*bsz

            # Calculate centroid
            centroid = class_features.mean(dim=0)
            new_protos[c] = centroid

        self.prototypes = torch.cat((self.prototypes, new_protos), dim=0)

    def adapt_prototypes(self, t, trn_loader, val_loader):
        self.model.eval()
        adapter = nn.Linear(self.S, self.S)
        if self.distiller_type == "mlp":
            adapter = nn.Sequential(nn.Linear(self.S, 2 * self.S),
                                      nn.GELU(),
                                      nn.Linear(2 * self.S, self.S)
                                      )
        adapter.to(self.device, non_blocking=True)
        optimizer, lr_scheduler = self.get_adapter_optimizer(adapter.parameters())
        old_prototypes = copy.deepcopy(self.prototypes)
        for epoch in range(self.nepochs // 2):
            adapter.train()
            train_loss, valid_loss = [], []
            for images, _ in trn_loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                with torch.no_grad():
                    target = self.model(images)
                    old_features = self.old_model(images)
                adapted_features = adapter(old_features)
                loss = torch.nn.functional.mse_loss(adapted_features, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), self.clipgrad)
                optimizer.step()
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()

            adapter.eval()
            with torch.no_grad():
                for images, _ in val_loader:
                    bsz = images.shape[0]
                    images = images.to(self.device, non_blocking=True)
                    target = self.model(images)
                    old_features = self.old_model(images)
                    adapted_features = adapter(old_features)
                    loss = torch.nn.functional.mse_loss(adapted_features, target)
                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} ")

        # Calculate new prototypes
        with torch.no_grad():
            adapter.eval()
            self.prototypes = adapter(self.prototypes)

            # Evaluation
            for (subset, loaders) in [("train", self.train_data_loaders), ("val", self.val_data_loaders)]:
                old_dist, new_dist = [], []
                class_images = np.concatenate([dl.dataset.images for dl in loaders])
                labels = np.concatenate([dl.dataset.labels for dl in loaders])

                for c in range(old_prototypes.shape[0]):
                    train_indices = torch.tensor(labels) == c

                    if isinstance(trn_loader.dataset.images, list):
                        train_images = list(compress(trn_loader.dataset.images, train_indices))
                        ds = ClassDirectoryDataset(train_images, val_loader.dataset.transform)
                    else:
                        ds = ClassMemoryDataset(class_images[train_indices], val_loader.dataset.transform)
                    loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)
                    from_ = 0
                    class_features = torch.full((2 * len(ds), self.S), fill_value=-999999999.0, device=self.device)
                    for images in loader:
                        bsz = images.shape[0]
                        images = images.to(self.device, non_blocking=True)
                        features = self.model(images)
                        class_features[from_: from_+bsz] = features
                        features = self.model(torch.flip(images, dims=(3,)))
                        class_features[from_+bsz: from_+2*bsz] = features
                        from_ += 2*bsz

                    # Calculate distance to old prototype
                    old_dist.append(torch.cdist(class_features, old_prototypes[c].unsqueeze(0)).mean())
                    new_dist.append(torch.cdist(class_features, self.prototypes[c].unsqueeze(0)).mean())

                old_dist = torch.stack(old_dist)
                new_dist = torch.stack(new_dist)
                print(f"Old {subset} distance: {old_dist.mean():.2f} ± {old_dist.std():.2f}")
                print(f"New {subset} distance: {new_dist.mean():.2f} ± {new_dist.std():.2f}")

    @torch.no_grad()
    def eval(self, t, val_loader):
        """ Perform nearest centroids classification """
        self.model.eval()
        tag_acc = Accuracy("multiclass", num_classes=self.prototypes.shape[0])
        taw_acc = Accuracy("multiclass", num_classes=self.classes_in_tasks[t])
        offset = self.task_offset[t]
        for images, target in val_loader:
            images = images.to(self.device, non_blocking=True)
            features = self.model(images)
            dist = torch.cdist(features, self.prototypes)
            tag_preds = torch.argmin(dist, dim=1)
            taw_preds = torch.argmin(dist[:, offset: offset + self.classes_in_tasks[t]], dim=1) + offset

            tag_acc.update(tag_preds.cpu(), target)
            taw_acc.update(taw_preds.cpu(), target)

        return 0, float(taw_acc.compute()), float(tag_acc.compute())

    def distill_knowledge(self, loss, features, distiller, old_features=None):
        if old_features is None:
            return loss, 0
        kd_loss = nn.functional.mse_loss(distiller(features), old_features)
        total_loss = (1 - self.alpha) * loss + self.alpha * kd_loss
        return total_loss, kd_loss

    def get_optimizer(self, parameters, wd):
        """Returns the optimizer"""
        milestones = (int(0.3*self.nepochs), int(0.6*self.nepochs), int(0.8*self.nepochs))
        optimizer = torch.optim.SGD(parameters, lr=self.lr, weight_decay=wd, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def get_adapter_optimizer(self, parameters, milestones=(40, 80)):
        """Returns the optimizer"""
        optimizer = torch.optim.SGD(parameters, lr=0.01, weight_decay=1e-5, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    @torch.no_grad()
    def check_singular_values(self, t, val_loader):
        self.model.eval()
        self.svals_explained_by.append([])
        for i, _ in enumerate(self.train_data_loaders):
            if isinstance(self.train_data_loaders[i].dataset.images, list):
                train_images = self.train_data_loaders[i].dataset.images
                ds = ClassDirectoryDataset(train_images, val_loader.dataset.transform)
            else:
                ds = ClassMemoryDataset(self.train_data_loaders[i].dataset.images, val_loader.dataset.transform)

            loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=val_loader.num_workers, shuffle=False)
            from_ = 0
            class_features = torch.full((len(ds), self.S), fill_value=-999999999.0, device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)
                class_features[from_: from_ + bsz] = features
                from_ += bsz

            cov = torch.cov(class_features.T)
            svals = torch.linalg.svdvals(cov)
            xd = torch.cumsum(svals, 0)
            xd = xd[xd < self.sval_fraction * torch.sum(svals)]
            explain = xd.shape[0]
            self.svals_explained_by[t].append(explain)

    def print_singular_values(self):
        print(f"{self.sval_fraction} of eigenvalues sum is explained by:")
        for t, explained_by in enumerate(self.svals_explained_by):
            print(f"Task {t}: {explained_by}")
