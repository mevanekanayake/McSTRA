from typing import Dict, Optional, Sequence, Tuple, Union
from utils.multicoil import rss
import numpy as np
import torch
from utils.mask import apply_mask, MaskFunc
from utils.fourier import fft, ifft
from utils.math import to_tensor, complex_center_crop, center_crop

class DataTransform:
    """
    Data Transformer for training CASTRA models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ):
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            image: Zero-filled image [1] 320, 320
            target: Target image [2] 320, 320
            kspace: Masked k-space [0] 2, 320, 320
            mask: Undersampling mask [3] 2, 320, 320
            fname: File name [4]
            slice_num: Serial number of the slice [5]
            max_value: Maximum value of the target VOLUME [6]
            mean_value: Mean value of the image [7]
            std_value: Standard Deviation value of the image [8]
            sequence: Sequence [9]
        """

        # convert to tensor
        kspace = to_tensor(kspace)
        target = to_tensor(target)

        # crop size
        crop_size = (320, 320)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # crop in image space
        image = ifft(kspace)
        image = complex_center_crop(image, crop_size)

        # normalize - NEW
        image = image / max_value
        target = target / max_value
        max_value = max_value / max_value

        target2 = image
        # get cropped kspace
        kspace = fft(image)
        kspace_ori = kspace
        # crop target
        target = center_crop(target, crop_size)

        # undersampling in kspace
        seed = None if not self.use_seed else tuple(map(ord, fname))
        kspace, mask = apply_mask(kspace, self.mask_func, seed)

        # zero-filled image
        image = ifft(kspace)
        if self.which_challenge == "multicoil":
            image = rss(image)

        # # normalize image
        # image, mean_value, std_value = normalize_instance(image)
        mean_value = 0
        std_value = 0
        # # normalize target
        # target = normalize(target, mean_value, std_value)

        # sequence
        sequence = attrs['acquisition']

        masked_psf = fft(mask)

        return image.permute(2, 0, 1), target.unsqueeze(0), kspace.permute(2, 0, 1), mask.permute(2, 0, 1), fname, slice_num, max_value, mean_value, std_value, sequence, target2.permute(2, 0, 1), masked_psf[160].permute(1, 0), kspace_ori.permute(2, 0, 1)


def make_ht_masks(num_hts, acc, w=320):

    n = num_hts
    ht_mask_list = []

    num_lines0 = int(2*np.ceil(w*acc[0]*2)/2)
    num_lines1 = int(w-num_lines0)

    num_lines1 = int(num_lines1/2)

    ht_mask_list.append(torch.cat([torch.zeros(w, num_lines1), torch.ones(w, num_lines0), torch.zeros(w, num_lines1)], dim=1))
    ht_mask_list.append(torch.cat([torch.ones(w, num_lines1), torch.zeros(w, num_lines0), torch.ones(w, num_lines1)], dim=1))

    if n == 3:
        ht_mask_list.append(torch.ones(w, w))

    return ht_mask_list


