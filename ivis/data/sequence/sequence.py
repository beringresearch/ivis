"""Common interface for all custom datasets compatible with ivis"""

from abc import abstractmethod
from collections.abc import Sequence

class IndexableDataset(Sequence):
    """A sequence that also defines a shape attribute. This indexable data structure
    can be provided as input to ivis."""
    @abstractmethod
    def shape(self):
        """Returns the shape of the dataset. First dimension corresponds to rows,
        the other dimensions correspond to features."""
        raise NotImplementedError

    def get_batch(self, idx_seq):
        """Returns a batch of data points based on the index sequence provided.

        Non-optimized version, can be overridden by child classes to be made be efficient"""
        return [self.__getitem__(item) for item in idx_seq]
