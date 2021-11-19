import torch
import torch.utils.data
from typing import Sized, Iterator
from torch.utils.data import Dataset, Sampler


class FirstLastSampler(Sampler):
    """
    A sampler that returns elements in a first-last order.
    """

    def __init__(self, data_source: Sized):
        """
        :param data_source: Source of data, can be anything that has a len(),
        since we only care about its number of elements.
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.indices = list(self)

    def __iter__(self) -> Iterator[int]:
        indices = []
        side = True
        steps = 0
        i = 0
        while i < len(self.data_source):
            indices.append(steps * (1 if side else -1))
            if side:
                steps += 1
            side = not side
            i += 1

        return iter(indices)

    def __len__(self):
        return len(self.data_source)


def create_train_validation_loaders(
        dataset: Dataset, validation_ratio, batch_size=100, num_workers=2
):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    # TODO:
    #  Create two DataLoader instances, dl_train and dl_valid.
    #  They should together represent a train/validation split of the given
    #  dataset. Make sure that:
    #  1. Validation set size is validation_ratio * total number of samples.
    #  2. No sample is in both datasets. You can select samples at random
    #     from the dataset.
    #  Hint: you can specify a Sampler class for the `DataLoader` instance
    #  you create.
    # ====== YOUR CODE: ======

    idx = len(dataset) - int((validation_ratio * len(dataset)))
    indices = list(range(len(dataset)))

    ds_train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices[:idx])
    ds_val_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices[idx:])

    dl_train = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, sampler=ds_train_sampler)

    dl_valid = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, sampler=ds_val_sampler)

    # ========================

    return dl_train, dl_valid
