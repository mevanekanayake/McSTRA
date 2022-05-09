from typing import Dict, Optional, Sequence, Tuple, Union
from utils_new.multicoil import rss
from utils_new.math import complex_abs
import numpy as np
from .mask import MaskFunc
import torch
from packaging import version
import math

if version.parse(torch.__version__) >= version.parse("1.7.0"):
    from utils_new.fourier import fft2c_new as fft2c
    from utils_new.fourier import ifft2c_new as ifft2c
else:
    from utils_new.fourier import fft2c_old as fft2c
    from utils_new.fourier import ifft2c_old as ifft2c


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    data = data.numpy()

    return data[..., 0] + 1j * data[..., 1]


def apply_mask(
        data: torch.Tensor,
        mask_func: MaskFunc,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data
            mask: The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1]:] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros
    mask = torch.ones_like(data) * mask + 0.0

    return masked_data, mask


def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        x:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask


def batched_mask_center(
        x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Can operate with different masks for each batch element.

    Args:
        x:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (
                not x.shape[0] == mask_to.shape[0]
        ):
            raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]


def center_crop_to_smallest(
        x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimum size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y


def normalize(
        data: torch.Tensor,
        mean: Union[float, torch.Tensor],
        stddev: Union[float, torch.Tensor],
        eps: Union[float, torch.Tensor] = 1e-11,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """

    if data.ndim == 4:
        output = (data - mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) / (stddev.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + eps)
    else:
        output = (data - mean) / (stddev + eps)

    return output


def unnormalize(
        data: torch.Tensor,
        mean: Union[float, torch.Tensor],
        stddev: Union[float, torch.Tensor],
        eps: Union[float, torch.Tensor] = 1e-11,
) -> torch.Tensor:
    if data.ndim == 4:
        output = (data * (stddev.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + eps)) + mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    else:
        output = (data * stddev) + mean

    return output


def normalize_instance(
        data: torch.Tensor, eps: Union[float, torch.Tensor] = 1e-11
) -> Tuple[torch.Tensor, Union[torch.Tensor], Union[torch.Tensor]]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """

    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


def normalize_batch(
        data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, Union[torch.Tensor], Union[torch.Tensor]]:
    """
    Normalize the given batch tensor with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed for each instance of the batch separately.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """

    mean = torch.mean(data, dim=(1, 2, 3)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    stddev = torch.std(data, dim=(1, 2, 3)).unsqueeze(1).unsqueeze(2).unsqueeze(3)

    return (data - mean) / (stddev + eps), mean, stddev


def get_target_and_image(kspace_raw, target_raw, attrs, mask_function, challenge, fname, use_seed=True):
    # convert to tensor

    kspace = to_tensor(kspace_raw)
    target = to_tensor(target_raw)

    # crop size
    crop_size = (320, 320)

    # check for max value
    max_value = attrs["max"] if "max" in attrs.keys() else 0.0

    # crop in image space
    image = ifft2c(kspace)
    image = complex_center_crop(image, crop_size)

    # normalize - NEW
    image = image / max_value
    target = target / max_value

    # get cropped kspace
    kspace = fft2c(image)

    # crop target
    target = center_crop(target, crop_size)

    # undersampling in kspace
    seed = None if not use_seed else tuple(map(ord, fname))
    kspace, mask = apply_mask(kspace, mask_function, seed)

    # zero-filled image
    image = ifft2c(kspace)
    image = complex_abs(image)
    if challenge == "multicoil":
        image = rss(image)

    return target, image


class UNetDataTransform:
    """
    Data Transformer for training U-Net models.
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
        image = ifft2c(kspace)
        image = complex_center_crop(image, crop_size)

        # normalize - NEW
        image = image / max_value
        target = target / max_value
        max_value = max_value / max_value

        # get cropped kspace
        kspace = fft2c(image)

        # crop target
        target = center_crop(target, crop_size)

        # undersampling in kspace
        seed = None if not self.use_seed else tuple(map(ord, fname))
        kspace, mask = apply_mask(kspace, self.mask_func, seed)

        # zero-filled image
        image = ifft2c(kspace)
        image = complex_abs(image)
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

        return image.unsqueeze(0), target.unsqueeze(0), kspace.permute(2, 0, 1), mask.permute(2, 0, 1), fname, slice_num, max_value, mean_value, std_value, sequence


class D5C5DataTransform:
    """
    Data Transformer for training D5C5 models.
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
        image = ifft2c(kspace)
        image = complex_center_crop(image, crop_size)

        # normalize - NEW
        image = image / max_value
        target = target / max_value
        max_value = max_value / max_value

        # get cropped kspace
        kspace = fft2c(image)

        # crop target
        target = center_crop(target, crop_size)

        # undersampling in kspace
        seed = None if not self.use_seed else tuple(map(ord, fname))
        kspace, mask = apply_mask(kspace, self.mask_func, seed)

        # zero-filled image
        image = ifft2c(kspace)
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

        return image.permute(2, 0, 1), target.unsqueeze(0), kspace.permute(2, 0, 1), mask.permute(2, 0, 1), fname, slice_num, max_value, mean_value, std_value, sequence


class MICCANDataTransform:
    """
    Data Transformer for training MICCAN models.
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
        image = ifft2c(kspace)
        image = complex_center_crop(image, crop_size)

        # normalize - NEW
        image = image / max_value
        target = target / max_value
        max_value = max_value / max_value
        # get cropped kspace
        kspace = fft2c(image)

        # crop target
        target = center_crop(target, crop_size)

        # undersampling in kspace
        seed = None if not self.use_seed else tuple(map(ord, fname))
        kspace, mask = apply_mask(kspace, self.mask_func, seed)

        # zero-filled image
        image = ifft2c(kspace)
        if self.which_challenge == "multicoil":
            image = rss(image)

        # normalize image
        # image, mean_value, std_value = normalize_instance(image)
        mean_value = 0
        std_value = 0
        # normalize target
        # target = normalize(target, mean_value, std_value)

        # sequence
        sequence = attrs['acquisition']

        return image.permute(2, 0, 1), target.unsqueeze(0), kspace.permute(2, 0, 1), mask.permute(2, 0, 1), fname, slice_num, max_value, mean_value, std_value, sequence


class ViTDataTransform:
    """
    Data Transformer for training ViT models.
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
        image = ifft2c(kspace)
        image = complex_center_crop(image, crop_size)

        # normalize - NEW
        image = image / max_value
        target = target / max_value
        max_value = max_value / max_value

        # get cropped kspace
        kspace = fft2c(image)

        # crop target
        target = center_crop(target, crop_size)

        # undersampling in kspace
        seed = None if not self.use_seed else tuple(map(ord, fname))
        kspace, mask = apply_mask(kspace, self.mask_func, seed)

        # zero-filled image
        image = ifft2c(kspace)
        image = complex_abs(image)
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

        return image.unsqueeze(0), target.unsqueeze(0), kspace.permute(2, 0, 1), mask.permute(2, 0, 1), fname, slice_num, max_value, mean_value, std_value, sequence


class CASTRADataTransform:
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
        image = ifft2c(kspace)
        image = complex_center_crop(image, crop_size)

        # normalize - NEW
        image = image / max_value
        target = target / max_value
        max_value = max_value / max_value

        target2 = image
        # get cropped kspace
        kspace = fft2c(image)
        kspace_ori = kspace
        # crop target
        target = center_crop(target, crop_size)

        # undersampling in kspace
        seed = None if not self.use_seed else tuple(map(ord, fname))
        kspace, mask = apply_mask(kspace, self.mask_func, seed)

        # zero-filled image
        image = ifft2c(kspace)
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

        masked_psf = fft2c(mask)

        return image.permute(2, 0, 1), target.unsqueeze(0), kspace.permute(2, 0, 1), mask.permute(2, 0, 1), fname, slice_num, max_value, mean_value, std_value, sequence, target2.permute(2, 0, 1), masked_psf[160].permute(1, 0), kspace_ori.permute(2, 0, 1)


def make_ht_masks(num_hts, acc, w=320):

    n = num_hts
    tri_n = n*(n+1)/2
    ht_mask_list = []

    # num_lines0 = int(2*np.ceil(w*acc[0]/2)) # 26
    # num_lines1 = w-num_lines0 # 294
    # num_lines0 = 64  # 1/5
    # num_lines1 = 256  # 4/5
    # # num_lines0 = 18  # 1/3
    # # num_lines1 = 302  # 2/3
    # num_lines0 = 54  # 1/6
    # num_lines1 = 266  # 5/6
    # num_lines0 = 48  # 1/6
    # num_lines1 = 272  # 5/6

    num_lines0 = int(2*np.ceil(w*acc[0]*2)/2)
    num_lines1 = int(w-num_lines0)

    num_lines1 = int(num_lines1/2)

    ht_mask_list.append(torch.cat([torch.zeros(w, num_lines1), torch.ones(w, num_lines0), torch.zeros(w, num_lines1)], dim=1))
    ht_mask_list.append(torch.cat([torch.ones(w, num_lines1), torch.zeros(w, num_lines0), torch.ones(w, num_lines1)], dim=1))

    if n == 3:
        ht_mask_list.append(torch.ones(w, w))

    # import matplotlib.pyplot as plt
    # for i, ht_mask in enumerate(ht_mask_list):
    #     ax = plt.imshow(ht_mask.detach().cpu(), cmap='gray', vmin=0., vmax=1.)
    #     ax.axes.get_xaxis().set_visible(False)
    #     ax.axes.get_yaxis().set_visible(False)
    #     ax = plt.show()
    #     # from pathlib import Path
    #     # figure_path = Path('D:\\TMI\\figures')
    #     # plt.savefig(f'{figure_path}/mask_ht_{i}.png', format='png', bbox_inches='tight', dpi=600)

    return ht_mask_list


