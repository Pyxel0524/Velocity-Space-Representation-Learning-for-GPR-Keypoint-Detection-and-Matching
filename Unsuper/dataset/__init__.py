import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from .MVGROUNDED import MVGROUNDED
from .MVCMU import MVCMU
from .MVNUDT import MVNUDT
from ..utils.common_utils import get_dist_info

__all__ = {
    'GROUNDED': MVGROUNDED,
    'CMU': MVCMU,
    'NUDT': MVNUDT
}


def build_dataloader(dataset_cfg, batch_size, dist, workers=4,
                     logger=None, training=True):
    dataset = __all__[dataset_cfg['data_name']](dataset_cfg)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    if training:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
            shuffle=(sampler is None) and training,
            drop_last=False, sampler=sampler, timeout=0
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
            shuffle=(sampler is None) and training,
            drop_last=False, sampler=sampler, timeout=0
        )

    return dataset, dataloader, sampler


def build_LightningDataloader(dataset_cfg, args, training=True):
    if training:
        dataset = __all__[dataset_cfg['train_name']](
            dataset_cfg,
            training
        )

        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                collate_fn=dataset.collate_batch)
    else:
        dataset = __all__[dataset_cfg['export_name']](
            dataset_cfg,
            training
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                collate_fn=dataset.test_collate_batch)
    return dataset, dataloader