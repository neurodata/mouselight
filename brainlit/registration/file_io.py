import numpy as np
import SimpleITK as sitk
from pathlib import Path
import pickle


def _validate_inputs(**kwargs):
    """Accepts arbitrary kwargs. If recognized, they are validated.
    If they cannot be validated an exception is raised.
    A dictionary containing the validated kwargs is returned."""

    # Validate data.

    if "data" in kwargs.keys():
        data = kwargs["data"]
        if isinstance(data, dict):
            # Verify that each value in the dict is a np.ndarray object and recursion will stop at 1 level.
            if not all(map(lambda datum: isinstance(datum, np.ndarray), data.values())):
                raise TypeError(
                    f"data must be either a np.ndarray or a dictionary mapping only to np.ndarray objects."
                )
            # Recurse into each element of the dictionary keys of np.ndarray objects.
            for key, datum in data.items():
                data[key] = _validate_inputs(data=datum)["data"]
        # If data is neither a list, dict, nor a np.ndarray, raise TypeError.
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"data must be a np.ndarray.\ntype(data): {type(data)}.")
        # data is a np.ndarray.
        kwargs.update(data=data)

    # Validate file_path.

    if "file_path" in kwargs.keys():
        file_path = kwargs["file_path"]
        file_path = (
            Path(file_path).expanduser().resolve()
        )  # Will raise exceptions if file_path is not an oppropriate type.
        if not file_path.parent.is_dir():
            raise FileNotFoundError(
                f"file_path corresponds to a location that does not presently exist.\n"
                f"file_path.parent: {file_path.parent}."
            )
        # Validate extension.
        if not file_path.suffix:
            default_extension = ".vtk"
            file_path = file_path.with_suffix(default_extension)
        # file_path is valid.
        kwargs.update(file_path=file_path)

    return kwargs


def save(data, file_path):
    """
    Save data to file_path

    Args:
        data (np.ndarray, dict): An array or a dictionary with np.ndarray values to be saved.
        file_path (str, Path): The file path that data will be saved to. Accepts an arbitrary suffix but provides .vtk by default.

    Raises:
        Exception: Raised if _validate_inputs has failed to catch an improper type for data.
    """

    # Validate inputs.
    inputs = {"data": data, "file_path": file_path}
    validated_inputs = _validate_inputs(**inputs)
    data = validated_inputs["data"]
    file_path = validated_inputs["file_path"]

    # data is either a single np.ndarray or a list of np.ndarrays.

    if isinstance(data, np.ndarray):
        # Convert data to sitk.Image.
        # Take the transpose of data to premptively correct for the f-ordering of SimpleITK Images.
        data = data.T
        data_Image = sitk.GetImageFromArray(data)  # Side effect: breaks alias.
        # Save data to file_path.
        sitk.WriteImage(data_Image, str(file_path))
    # If data is a dictionary it must map to np.ndarray objects.
    elif isinstance(data, dict):
        np.savez(file_path.with_suffix(""), **data)  # '.npz' is appended.
    else:
        # _validate_inputs has failed.
        raise Exception(
            f"_validate_inputs has failed to prevent an improper type for data.\n"
            f"type(data): {type(data)}."
        )


def load(file_path):
    """
    Load data from file_path.

    Args:
        file_path (str, Path): The file path from which data will be retrieved.

    Returns:
        np.ndarray, dict: The array or dict of arrays saved at file_path.
    """

    # Validate inputs.
    inputs = {"file_path": file_path}
    validated_inputs = _validate_inputs(**inputs)
    file_path = validated_inputs["file_path"]

    if file_path.suffix == ".npz":
        data = np.load(file_path)
        # data is a dictionary.
        return data
    else:
        # Read in data as sitk.Image.
        data_Image = sitk.ReadImage(str(file_path))
        # Convert data_Image to np.ndarray.
        data = sitk.GetArrayFromImage(data_Image)
        # Take the transpose of data to correct for the f-ordering of SimpleITK Images.
        data = data.T
        # data is a np.ndarray.
        return data


def save_pickled(obj, file_path):
    """
    Pickle object obj and save it to file_path.

    Args:
        obj (object): The pickleable object to be saved.
        file_path (str, Path): The file path at which to save obj.
    """

    # Validate file_path.
    file_path = _validate_inputs(file_path=file_path)["file_path"]

    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def load_pickled(file_path):
    """
    Load pickled object from file_path.

    Args:
        file_path (str, Path): The file path at which a pickled object is saved.

    Returns:
        object: The pickled object saved at file_path.
    """

    # Validate file_path.
    file_path = _validate_inputs(file_path=file_path)["file_path"]

    # TODO: verify behavior of suffixes.
    with open(file_path, "rb") as file:
        return pickle.load(file)
