import numpy as np

from ..lddmm._lddmm_utilities import _validate_ndarray


def _verify_data_is_ndarray(data):
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be of type np.ndarray.\ntype(data): {type(data)}.")


def cast_to_typed_array(data, dtype=float):
    """
    Returns a copy of data cast as a np.ndarray of type dtype.

    Args:
        data (np.ndarray): The array to be cast.
        dtype (type, optional): The dtype to cast data to. Defaults to float. Defaults to float.

    Returns:
        np.ndarray: A copy of data cast to type dtype.
    """

    return _validate_ndarray(data, dtype=dtype)


def normalize_by_MAD(data):
    """
    Returns a copy of data divided by its mean absolute deviation.

    Args:
        data (np.ndarray): The array to be normalized.

    Returns:
        np.ndarray: A copy of data divided by its mean absolute deviation.
    """

    _verify_data_is_ndarray(data)

    mean_absolute_deviation = np.mean(np.abs(data - np.median(data)))

    normalized_data = data / mean_absolute_deviation

    return normalized_data


def center_to_mean(data):
    """
    Returns a copy of data subtracted by its mean.

    Args:
        data (np.ndarray): The array to be subtracted from.
    Returns:
        np.ndarray: A copy of data subtracted by its mean.
    """

    _verify_data_is_ndarray(data)

    centered_data = data - np.mean(data)

    return centered_data


def pad(data, pad_width=10, mode="constant", constant_values=None):
    """
    Returns a padded copy of data.

    Args:
        data (np.ndarray): The array to be padded.
        pad_width (int, optional): The amount by which to pad. Defaults to 10.
        mode (str, optional): The padding mode used in np.pad. Defaults to 'constant'.
        constant_values (float, optional): The values to use in padding if mode='constant' If None, this is set to np.quantile(data, 10**-data.ndim). Defaults to None.

    Returns:
        np.ndarray: The padded copy of data.
    """

    _verify_data_is_ndarray(data)

    if constant_values is None:
        constant_values = np.quantile(data, 10 ** -data.ndim)

    pad_kwargs = {"array": data, "pad_width": pad_width, "mode": mode}
    if mode == "constant":
        pad_kwargs.update(constant_values=constant_values)

    padded_data = np.pad(**pad_kwargs)

    return padded_data
