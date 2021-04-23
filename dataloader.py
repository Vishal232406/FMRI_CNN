from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


import nibabel as nib
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

# nii paths
IMGS: List[Path] = sorted(Path(__file__).resolve().parent.rglob("*.nii"))
# Path to a custom csv file with the file name, subject id, and diagnosis
ANNOTATIONS: DataFrame = pd.read_csv(Path(__file__).resolve().parent / "NEW_FMRI.csv")
FmriSlice = Tuple[int, int, int, int]  # just a convencience type to save space


class RandomFmriDataset(Dataset):
    """Just grabs a random patch of size `patch_shape` from a random brain.

    Parameters
    ----------
    patch_shape: Tuple[int, int, int, int]
        The patch size.

    standardize: bool = True
        Whether or not to do intensity normalization before returning the Tensor.

    transform: Optional[Callable] = None
        The transform to apply to the 4D array.

   """
    def __init__(
        self,
        patch_shape: Optional[FmriSlice] = None,
        standardize: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        self.annotations = ANNOTATIONS
        self.img_paths = IMGS
        self.labels: List[int] = []
        self.shapes: List[Tuple[int, int, int, int]] = []
        for img in IMGS:  # get the diagnosis, 0 = control, 1 = autism and other info
            file_id = img.stem.replace("_func_minimal", "")
            label_idx = self.annotations["FILE_ID"] == file_id
            self.labels.append(self.annotations.loc[label_idx, "DX_GROUP"])
            self.shapes.append(nib.load(str(img)).shape)  # usually (61, 73, 61, 236)
        self.max_dims = np.max(self.shapes, axis=0)
        self.min_dims = np.min(self.shapes, axis=0)

        self.standardize = standardize
        self.transform = transform

        # ensure patch shape is valid
        if patch_shape is None:
            smallest_dims = np.min(self.shapes, axis=0)[:-1]  # exclude time dim
            self.patch_shape = (*smallest_dims, 8)
        else:
            if len(patch_shape) != 4:
                raise ValueError("Patches must be 4D for fMRI")
            for dim, max_dim in zip(patch_shape, self.max_dims):
                if dim > max_dim:
                    raise ValueError("Patch size too large for data")
            self.patch_shape = patch_shape

    def __len__(self) -> int:
        # when generating the random dataloader, the "length" is kind of phoney. You could make the
        # length be anything, e.g. 1000, 4962, or whatever. However, what you set as the length will
        # define the epoch size.
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tensor:
        # just return a random patch
        global array_1
        path = np.random.choice(self.img_paths)
        img = nib.load(str(path))
        # going larger than max_idx will put us past the end of the array
        max_idx = np.array(img.shape) - np.array(self.patch_shape) + 1

        # Python has a `slice` object which you can use to index into things with the `[]` operator
        # we are going to build the slices we need to index appropriately into our niis with the
        # `.dataobj` trick
        slices = []
        for length, maximum in zip(self.patch_shape, max_idx):
            start = np.random.randint(0, maximum)
            slices.append(slice(start, start + length))
        array = img.dataobj[slices[0], slices[1], slices[2], slices[3]]

        if self.standardize:
            array_1 = np.copy(array)
            array_1 -= np.mean(array_1)
            array_1 /= np.std(array_1, ddof=1)
        return torch.Tensor(array_1)

    def test_get_item(self) -> None:
        """Just test that the produced slices can't ever go past the end of a brain"""
        for path in self.img_paths:
            img = nib.load(str(path))
            max_idx = np.array(img.shape) - np.array(self.patch_shape) + 1
            max_dims = img.shape
            for length, maximum, max_dim in zip(self.patch_shape, max_idx, max_dims):
                for start in range(maximum):
                    # array[a:maximum] is to the end
                    assert start + length <= max_dim
                    if start == maximum - 1:
                        assert start + length == max_dim


