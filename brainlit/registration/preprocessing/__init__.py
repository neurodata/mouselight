from .normalization import cast_to_typed_array
from .normalization import normalize_by_MAD
from .normalization import center_to_mean
from .normalization import pad

from .bias_and_artifact_correction import correct_bias_field
from .bias_and_artifact_correction import remove_grid_artifact

from .initialization import locally_rotate_velocity_fields
from .initialization import locally_translate_velocity_fields


# TODO: update preprocessing_functions, include resample.
preprocessing_functions = [
    # from .normalization:
    "cast_to_typed_array",
    "normalize_by_MAD",
    "center_to_mean",
    "pad",
    # from .bias_and_artifact_correction:
    "correct_bias_field",
    "remove_grid_artifact",
    # from .initialization:
    "locally_rotate_velocity_fields",
    "locally_translate_velocity_fields",
]

"""
Preprocessing Pipeline
"""

from collections.abc import Iterable

import numpy as np

from ..lddmm._lddmm_utilities import _validate_ndarray


def preprocess(data, processes):
    """
    Perform each preprocessing function in processes, in the order listed,
    on data if it is an array, or on each element in data if it is a list of arrays.

    Args:
        data (np.ndarray, list): The array or list of arrays to be preprocessed.
        processes (list): The list of strings, each corresponding to the name of a preprocessing function.
        process_kwargs (seq, optional): A sequence of dictionaries containing kwargs for each element of processes. Defaults to None.

    Raises:
        TypeError: Raised if data is a list whose elements are not all of type np.ndarray.
        TypeError: Raised if data is neither a np.ndarray or a list of np.ndarrays.
        TypeError: Raised if an element of processes is neither a single string nor an iterable.
        ValueError: Raised if an element of processes is an iterable but not of length 2.
        ValueError: Raised if an alement of processes is a 2-element iterable whose first element is not a string.
        ValueError: Raised if an alement of processes is a 2-element iterable whose second element is not a dictionary.
        TypeError: Raised if an element of processes includes a dictionary with a key that is not a string.
        ValueError: Raised if any element of processes is not a recognized preprocessing function.

    Returns:
        np.ndarray, list: A copy of data after having each function in processes applied.
    """

    # Check data form to match output.
    data_given_as_list = isinstance(data, list)

    # Validate inputs.

    # Verify data.
    if isinstance(data, list):
        if not all(isinstance(datum, np.ndarray) for datum in data):
            raise TypeError(
                f"If data is a list, all elements must be np.ndarrays.\n"
                f"type(data[0]): {type(data[0])}."
            )
    elif isinstance(data, np.ndarray):
        data = [data]
    else:
        # data is neither a list nor a np.ndarray.
        raise TypeError(f"data must be a np.ndarray or a list of np.ndarrays.")

    # Validate processes.
    processes = _validate_ndarray(processes, required_ndim=1)
    for process_index, process in enumerate(processes):
        # Check whether this process is just a single string.
        if isinstance(process, str):
            continue
        # This process is not a single string.
        # Check whether it is an Iterable.
        if not isinstance(process, Iterable):
            raise TypeError(
                f"If an element of processes is not a single string then it must be an iterable.\n"
                f"type(processes[{process_index}]: {type(process)}."
            )
        # This process is an Iterable.
        # Check whether this process has length 2.
        if len(process) != 2:
            raise ValueError(
                f"If an element of processes is not a single string then it must be a 2-element iterable.\n"
                f"type(processes[{process_index}]): {type(process)}.\n"
                f"len(processes[{process_index}]): {len(process)}."
            )
        # This process is an Iteraable of length 2.
        # Check whether the first element of this process is a string.
        if not isinstance(process[0], str):
            raise ValueError(
                f"If an element of processes is a 2-element iterable, the first element must be of type str.\n"
                f"type(processes[{process_index}][0]: {type(process[0])}."
            )
        # Check whether the second element of this process is a dict.
        if not isinstance(process[1], dict):
            raise ValueError(
                f"If an element of processes is a 2-element iterable, the second element must be of type dict.\n"
                f"type(processes[{process_index}][1]: {type(process[1])}."
            )
        # Check whether all keys of the process[1] dictionary are strings.
        for key_index, key in enumerate(process[1].keys()):
            if not isinstance(key, str):
                raise TypeError(
                    f"Each dictionary of kwargs must all keys be of type str.\n"
                    f"type(processes[{process_index}][1].keys()[{key_index}]): {type(key)}."
                )

    # Process each np.ndarray.
    # If data was passed in as a single np.ndarray,
    # then data is now a 1-element list containing that np.ndarray: [data].
    for data_index, datum in enumerate(data):
        for process_index, process in enumerate(processes):
            # process is either a string or a 2-element sequence whose first element is a string and whose second element is a dictionary of kwargs for that process.
            if isinstance(process, str):
                process, process_kwargs = process, dict()
            else:
                process, process_kwargs = process
            # process is a string indicating a preprocessing function, process_kwargs is a dictionary of kwargs to be passed to that process.
            if process in preprocessing_functions:
                datum = eval(f"{process}(datum, **process_kwargs)")
            else:
                raise ValueError(
                    f"Process {process} not recognized.\n"
                    f"Recognized processes: {preprocessing_functions}."
                )
        data[data_index] = datum

    # Return in a form appropriate to what was passed in,
    # i.e. list in, list out, np.ndarray in, np.ndarray out.
    return data if data_given_as_list else data[0]
