import numpy as np
import torch


class TabularDataLoader:
    """
    Data loader for tabular data.
    Significantly faster than the PyTorch DataLoader (for tabular data).
    """

    def __init__(
        self, X: np.ndarray[float], y: np.ndarray[float], batch_size: int, shuffle: bool
    ):
        """
        Initialises the data loader

        Parameters
        ----------
        X: np.ndarray[float]
            The input data
        y: np.ndarray[float]
            The target data
        batch_size: int
            The batch size
        shuffle: bool
            Whether to shuffle the data between iterations (epochs)
        """
        self.X = torch.from_numpy(X).to(torch.float32)
        self.y = torch.from_numpy(y).to(torch.float32)

        assert (
            self.X.shape[0] == self.y.shape[0]
        ), "X and y must have the same number of samples"

        self.batch_size = batch_size
        self.shuffle = shuffle

        self._i = 0

    @property
    def n_samples(self):
        """Number of samples"""
        return self.X.shape[0]

    @property
    def n_batches(self):
        """Number of batches"""
        return int(np.ceil(self.n_samples // self.batch_size))

    def __iter__(self):
        """Initialises the iterator"""
        if self.shuffle:
            self.indices = np.random.permutation(self.n_samples)
        else:
            self.indices = np.arange(self.n_samples)
        self._i = 0
        return self

    def __len__(self) -> int:
        """Returns the number of batches"""
        return self.n_batches

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, np.ndarray[int]]:
        """
        Returns the next batch

        Returns
        -------
        batch_X: torch.Tensor
            The input data
        batch_y: torch.Tensor
            The target data
        batch_ind: np.ndarray[int]
            The indices of the samples in the batch
        """
        if self._i >= self.n_samples:
            raise StopIteration

        # Get the batch indices
        batch_ind = self.indices[
            self._i : min(self._i + self.batch_size, self.n_samples)
        ]
        self._i += self.batch_size

        # Get the batch data
        batch_X = self.X[batch_ind]
        batch_y = self.y[batch_ind]

        return batch_X, batch_y, batch_ind
