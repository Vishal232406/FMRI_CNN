from typing import Tuple

import numpy as np

from dataloader import RandomFmriDataset, FmriDataset
import matplotlib.pyplot as plt

FmriSlice = Tuple[int, int, int, int]  # just a convencience type to save space

def plot_patch(size: Tuple[int, int, int, int] = (32, 32, 32, 12)) -> None:
    dl = RandomFmriDataset(size)
    img = dl[0].numpy()
    img -= np.mean(img)
    img /= np.std(img, ddof=1)
    img = np.clip(img, -5, 5)
    fig, axes = plt.subplots(nrows=1, ncols=3)
    axes.flat[0].imshow(img[:, :, 0, 0], cmap="Greys")
    axes.flat[1].imshow(img[:, 0, :, 0], cmap="Greys")
    axes.flat[2].imshow(img[0, :, :, 0], cmap="Greys")
    plt.show()


def test_random_loading() -> None:
    patch_shape = (48, 48, 48, 12)
    dl = RandomFmriDataset(patch_shape)
    dl.test_get_item()


def test_mapped_loading(patch_shape: FmriSlice = (32, 32, 32, 12), strides: FmriSlice = (2, 2, 2, 4)) -> None:
    ds = FmriDataset(patch_shape=patch_shape, strides=strides)
    print(ds.n_patches)
    ds.test_get_item()


if __name__ == "__main__":
    # test_random_loading()
    # plot_patch((48, 48, 48, 12))
    # test_mapped_loading(patch_shape=(32, 32, 32, 12), strides=(16, 16, 16, 6))  # for "fast" testing
    plot_patch()
