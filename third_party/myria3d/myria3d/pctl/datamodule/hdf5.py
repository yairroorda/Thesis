from numbers import Number
from typing import Callable, Dict, List, Optional

from numpy.typing import ArrayLike
from pytorch_lightning import LightningDataModule
from torch_geometric.data import Data

from myria3d.pctl.dataloader.dataloader import GeometricNoneProofDataloader
from myria3d.pctl.dataset.iterable import InferenceDataset
from myria3d.pctl.dataset.utils import pre_filter_below_n_points
from myria3d.pctl.transforms.compose import CustomCompose
from myria3d.utils import utils

log = utils.get_logger(__name__)

TRANSFORMS_LIST = List[Callable]


class HDF5LidarDataModule(LightningDataModule):
    """Inference-only datamodule for prediction."""

    def __init__(
        self,
        epsg: str,
        points_pre_transform: Optional[Callable[[ArrayLike], Data]] = None,
        pre_filter: Optional[Callable[[Data], bool]] = pre_filter_below_n_points,
        tile_width: Number = 1000,
        subtile_width: Number = 50,
        subtile_overlap_predict: Number = 0,
        batch_size: int = 12,
        num_workers: int = 1,
        prefetch_factor: int = 2,
        transforms: Optional[Dict[str, TRANSFORMS_LIST]] = None,
        **kwargs,
    ):
        super().__init__()

        self.epsg = epsg
        self.points_pre_transform = points_pre_transform
        self.pre_filter = pre_filter

        self.tile_width = tile_width
        self.subtile_width = subtile_width
        self.subtile_overlap_predict = subtile_overlap_predict

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        t = transforms or {}
        self.preparation_predict_transform: TRANSFORMS_LIST = t.get("preparations_list", [])
        self.normalization_transform: TRANSFORMS_LIST = t.get("normalizations_list", [])

    @property
    def predict_transform(self) -> CustomCompose:
        return CustomCompose(self.preparation_predict_transform + self.normalization_transform)

    def _set_predict_data(self, las_file_to_predict):
        self.predict_dataset = InferenceDataset(
            las_file_to_predict,
            self.epsg,
            points_pre_transform=self.points_pre_transform,
            pre_filter=self.pre_filter,
            transform=self.predict_transform,
            tile_width=self.tile_width,
            subtile_width=self.subtile_width,
            subtile_overlap=self.subtile_overlap_predict,
        )

    def predict_dataloader(self):
        return GeometricNoneProofDataloader(
            dataset=self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=1,  # always 1 because this is an iterable dataset
            prefetch_factor=self.prefetch_factor,
        )
