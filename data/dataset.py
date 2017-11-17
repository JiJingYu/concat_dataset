import bisect
import numpy as np

class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class TensorDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


class H5Dataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.shape[0] == target_tensor.shape[0]
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        # print(index)
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]


class H5Imageset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): H,W,C.
        target_tensor (Tensor): H,W,C.
    """

    def __init__(self, data_tensor, target_tensor, patch_size, stride):
        assert data_tensor.shape[0] == target_tensor.shape[0]
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        ## idx_list的生成方法为该类的核心，可自行根据需要替换为其他函数。
        self.idx_list = self.get_idx_list(patch_size, stride)

    def get_idx_list(self, patch_size, stride):
        """
        idx_list的生成方法为该类的核心，可自行根据需要替换为其他函数。
        """
        H, W, _ = self.data_tensor.shape
        idx_list = []
        for h in np.arange(start=0, stop=H-patch_size, step=stride):
            for w in np.arange(start=0, stop=W - patch_size, step=stride):
                idx_list.append((h, w, patch_size, patch_size))
        return idx_list

    def __getitem__(self, index):
        # print(index)
        h, w, patch_size, patch_size = self.idx_list[index]
        return self.data_tensor[h:h+patch_size, w:w+patch_size], \
               self.target_tensor[h:h+patch_size, w:w+patch_size]

    def __len__(self):
        return len(self.idx_list)


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (iterable): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cummulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cummulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cummulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cummulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]
