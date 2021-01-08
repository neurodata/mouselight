from ..preprocessing import preprocess


def basic_preprocessing(data):
    """
    Call each of the listed preprocessing functions on <data> and return the result.
    Preprocessing functions:

        - cast_to_typed_array
        - normalize_by_MAD
        - center_to_mean

    Args:
        data (np.ndarray, list): An array to be preprocessed, or a list of arrays to be preprocessed.

    Returns:
        np.ndarray, list: The input array <data> after preprocessing, or each preprocessed array from the input list <data> if a list of arrays was provided.
    """

    return preprocess(
        data, ["cast_to_typed_array", "normalize_by_MAD", "center_to_mean"]
    )


def basic_preprocessing_with_pad(data):
    """
    Call each of the listed preprocessing functions on <data> and return the result.
    Preprocessing functions:

        - cast_to_typed_array
        - pad
        - normalize_by_MAD
        - center_to_mean

    Args:
        data (np.ndarray, list): An array to be preprocessed, or a list of arrays to be preprocessed.

    Returns:
        np.ndarray, list: The input array <data> after preprocessing, or each preprocessed array from the input list <data> if a list of arrays was provided.
    """

    return preprocess(
        data, ["cast_to_typed_array", "pad", "normalize_by_MAD", "center_to_mean"]
    )
