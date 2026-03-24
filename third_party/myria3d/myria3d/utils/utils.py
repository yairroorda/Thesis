import logging
import time
import warnings
from typing import Sequence

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.debug("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "task",
        "seed",
        "logger",
        "datamodule",
        "dataset_description",
        "predict",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    import rich
    import rich.syntax
    import rich.tree

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


def eval_time(method):
    """Decorator to log the duration of the decorated method"""

    def timed(*args, **kwargs):
        log = get_logger()
        time_start = time.time()
        result = method(*args, **kwargs)
        time_elapsed = round(time.time() - time_start, 2)

        log.info(f"Runtime of {method.__name__}: {time_elapsed}s")
        return result

    return timed


def define_device_from_config_param(gpus_param):
    """
    Param can be an in specifying a number of GPU to use (0 or 1) or an int in
    a list specifying which GPU to use (cuda:0, cuda:1, etc.)
    """
    device = torch.device("cpu") if gpus_param == 0 else (torch.device("cuda") if gpus_param == 1 else f"cuda:{int(gpus_param[0])}")
    return device
