import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional, Union, Callable, Type

from models.base import BaseLearner
from utils.inc_net import get_convnet
from utils.Buffer import RandomBuffer, activation_t
from convs.linears import AnalyticLinear, RecursiveLinear




class AnalyticDSAL(torch.nn.Module):
    """Dual-Stream Analytic Learning head (main + compensation streams)."""

    def __init__(
        self,
        backbone_output: int,
        backbone: Callable[[torch.Tensor], torch.Tensor] = torch.nn.Flatten(),
        expansion_size: int = 8192,
        gamma_main: float = 1e-3,
        gamma_comp: float = 1e-3,
        C: float = 1,
        activation_main: activation_t = torch.relu,
        activation_comp: activation_t = torch.tanh,
        device=None,
        dtype=torch.double,
        linear: Type[AnalyticLinear] = RecursiveLinear,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = backbone
        self.expansion_size = expansion_size
        self.buffer = RandomBuffer(
            backbone_output,
            expansion_size,
            activation=torch.nn.Identity(),
            **factory_kwargs,
        )
        self.activation_main = activation_main
        self.activation_comp = activation_comp
        self.C = C
        self.main_stream = linear(expansion_size, gamma_main, **factory_kwargs)
        self.comp_stream = linear(expansion_size, gamma_comp, **factory_kwargs)
        self.eval()

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.buffer(self.backbone(X))
        X_main = self.main_stream(self.activation_main(X))
        X_comp = self.comp_stream(self.activation_comp(X))
        return X_main + self.C * X_comp

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor, increase_size: int) -> None:
        num_classes = max(self.main_stream.out_features, int(y.max().item()) + 1)
        Y_main = torch.nn.functional.one_hot(y, num_classes=num_classes)
        X = self.buffer(self.backbone(X))

        X_main = self.activation_main(X)
        self.main_stream.fit(X_main, Y_main)
        self.main_stream.update()

        Y_comp = Y_main - self.main_stream(X_main)
        Y_comp[:, :-increase_size] = 0

        X_comp = self.activation_comp(X)
        self.comp_stream.fit(X_comp, Y_comp)

    @torch.no_grad()
    def update(self) -> None:
        self.main_stream.update()
        self.comp_stream.update()


class DSALNet(nn.Module):
    """Wrapper that combines a convolutional backbone with the analytic DSAL head."""

    def __init__(self, backbone: nn.Module, dsal_module: AnalyticDSAL):
        super().__init__()
        self.convnet = backbone
        self.dsal = dsal_module

    @property
    def feature_dim(self) -> int:
        return self.convnet.out_dim

    def extract_vector(self, x: torch.Tensor) -> torch.Tensor:
        return self.convnet(x)["features"]

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.convnet(x)["features"]
        feats = torch.nn.functional.normalize(feats, dim=1)
        logits = self.dsal(feats)
        return {"logits": logits, "features": feats}


class DSAL(BaseLearner):
    """Dual-Stream Analytic Learning (DS-AL) integration for the HRRP framework."""

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = self._build_network(args)

        # Hyper-parameters with safe fallbacks
        self.gamma_main = args.get("gamma", args.get("gamma_main", 1e-3))
        self.gamma_comp = args.get("gamma_comp", 1e-3)
        self.comp_ratio = args.get("compensation_ratio", 1.0)
        self.expansion_size = args.get("buffer_size", args.get("expansion_size", 8192))
        self.activation_main = torch.relu
        self.activation_comp = torch.tanh

    def _build_network(self, args) -> DSALNet:
        # Backbone for feature extraction (shared across tasks)
        if not self.args["init_train"]:
            backbone = torch.load(self.args["convnet_path"])
        else:
            backbone = get_convnet(args, pretrained=False)
        # Analytic head operates on backbone features directly
        dsal_module = AnalyticDSAL(
            backbone_output=backbone.out_dim,
            backbone=torch.nn.Identity(),
            expansion_size=args.get("expansion_size", 8192),
            gamma_main=args.get("gamma", args.get("gamma_main", 1e-3)),
            gamma_comp=args.get("gamma_comp", 1e-3),
            C=args.get("compensation_ratio", 1.0),
            activation_main=torch.relu,
            activation_comp=torch.tanh,
            device=self._device,
            dtype=torch.float,
        )
        return DSALNet(backbone, dsal_module)

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        increment_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + increment_size

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # Datasets
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"],
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args["num_workers"],
        )

        # Move to device and freeze backbone weights (analytic head is non-gradient)
        self._network.to(self._device)
        self._network.convnet.eval()

        # Fit DSAL incrementally using mini-batches
        for _, inputs, targets in train_loader:
            inputs = inputs.to(self._device)
            targets = targets.to(dtype = torch.int64, device=self._device)
            with torch.no_grad():
                feats = self._network.convnet(inputs)["features"]
                feats = torch.nn.functional.normalize(feats, dim=1)
            self._network.dsal.fit(feats, targets, increase_size=increment_size)

        self._network.dsal.update()

    def _compute_accuracy(self, model, loader):
        return super()._compute_accuracy(model, loader)
