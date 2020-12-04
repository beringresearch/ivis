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
