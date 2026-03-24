import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import knn_interpolate

from myria3d.models.modules.pyg_randla_net import PyGRandLANet
from myria3d.utils import utils

log = utils.get_logger(__name__)

MODEL_ZOO = [PyGRandLANet]


def get_neural_net_class(class_name: str) -> nn.Module:
    """A Class Factory to class of neural net based on class name.

    :meta private:

    Args:
        class_name (str): the name of the class to get.

    Returns:
        nn.Module: CLass of requested neural network.
    """
    for neural_net_class in MODEL_ZOO:
        if class_name in neural_net_class.__name__:
            return neural_net_class
    raise KeyError(f"Unknown class name {class_name}")


class Model(LightningModule):
    """Inference-only Lightning module for checkpoint-based prediction."""

    def __init__(self, **kwargs):
        """Initialization method of the Model lightning module.

        Everything needed to train/evaluate/test/predict with a neural architecture, including
        the architecture class name and its hyperparameter.

        See config files for a list of kwargs.

        """
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(ignore=["criterion"])

        neural_net_class = get_neural_net_class(kwargs.get("neural_net_class_name"))
        self.model = neural_net_class(**kwargs.get("neural_net_hparams"))

    def forward(self, batch: Batch) -> torch.Tensor:
        """Forward pass of neural network.

        Args:
            batch (Batch): Batch of data including x (features), pos (xyz positions),
            and y (targets, optionnal) in (B*N,C) format.

        Returns:
            torch.Tensor (B*N,1): targets
            torch.Tensor (B*N,C): logits

        """
        logits = self.model(batch.x, batch.pos, batch.batch, batch.ptr)
        if "copies" not in batch:
            raise RuntimeError("Inference expects copied full/sampled positions in batch.copies.")

        # During evaluation on test data and inference, we interpolate predictions back to original positions
        # KNN is way faster on CPU than on GPU by a 3 to 4 factor.
        logits = logits.cpu()
        batch_y = self._get_batch_tensor_by_enumeration(batch.idx_in_original_cloud)
        logits = knn_interpolate(
            logits.cpu(),
            batch.copies["pos_sampled_copy"].cpu(),
            batch.copies["pos_copy"].cpu(),
            batch_x=batch.batch.cpu(),
            batch_y=batch_y.cpu(),
            k=self.hparams.interpolation_k,
            num_workers=self.hparams.num_workers,
        )
        targets = None  # no targets in inference mode.
        if "transformed_y_copy" in batch.copies:
            # eval (test/val).
            targets = batch.copies["transformed_y_copy"].to(logits.device)
        return targets, logits

    def predict_step(self, batch: Batch) -> dict:
        """Prediction step.

        Move to CPU to avoid acucmulation of predictions into gpu memory.

        Args:
            batch (torch_geometric.data.Batch): Batch of data including x (features), pos (xyz positions),
            and y (targets, optionnal) in (B*N,C) format.

        Returns:
            dict: Dictionnary with predicted logits as well as input batch.

        """
        _, logits = self.forward(batch)
        return {"logits": logits.detach().cpu()}

    def _get_batch_tensor_by_enumeration(self, pos_x: torch.Tensor) -> torch.Tensor:
        """Get batch tensor (e.g. [0,0,1,1,2,2,...,B-1,B-1] )
        from shape B,N,... to shape (N,...).
        """
        return torch.cat([torch.full((len(sample_pos),), i) for i, sample_pos in enumerate(pos_x)])
