import os

from torch.utils.data import DataLoader


def worker_init_fn(worker_id):
    os.sched_setaffinity(0, list(range(os.cpu_count())))


def get_dataloader(
    dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    collate_fn=None,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
        drop_last=drop_last,
    )
