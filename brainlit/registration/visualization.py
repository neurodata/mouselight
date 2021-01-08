import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn.plotting as niplot


def _scale_data(data, limit_mode=None, stdevs=4, quantile=0.01, limits=None):
    """Returns a copy of data scaled such that the bulk of the values are mapped to the range [0, 1].

    Upper and lower limits are chosen based on limit_mode and other arguments.
    These limits are the anchors by which <data> is scaled such that after scaling,
    they lie on either end of the range [0, 1].

    supported limit_mode values:
    - 'valid mode' | related_kwarg [default kwarg value] --> description
    - 'stdev' | stdevs [4] --> limits are the mean +/- <stdevs> standard deviations
    - 'quantile' | quantile [0.05] --> limits are the <quantile> and 1 - <quantile> quantiles

    If <limits> is provided as a 2-element iterable, it will override limit_mode
    and be used directly as the anchoring limits:
    <limits> = (lower, upper)."""

    # Verify type of data.
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be of type np.ndarray.\ntype(data): {type(data)}.")

    # Determine scaling limits.

    if limits is not None:
        # TODO: make <limits> validation robust.
        if not isinstance(limits, (tuple, list, np.ndarray)):
            raise TypeError(
                f"If provided, limits must be one of the following types: tuple, list, np.ndarray.\n"
                f"type(limits): {type(limits)}."
            )
        else:
            # <limits> is a tuple, list, or np.ndarray.
            try:
                if len(limits) != 2:
                    raise ValueError(
                        f"If provided, limits must have length 2.\n"
                        f"len(limits): {len(limits)}."
                    )
            except TypeError:
                raise ValueError(
                    f"limits was provided as a 0-dimensional np.ndarray. It must have length 2.\n"
                    f"limits: {limits}."
                )
        # <limits> is a 2-element tuple, list, or np.ndarray.
        lower_lim, upper_lim = limits
    else:
        # limits is None. Use limit_mode to determine upper and lower limits.
        # List supported limit_mode values.
        supported_limit_modes = [None, "stdev", "quantile"]
        # Handle default None value, with no meaningful limits.
        if limit_mode is None:
            lower_lim = np.min(data)
            upper_lim = np.max(data)
        # Check limit_mode type.
        elif not isinstance(limit_mode, str):
            raise TypeError(
                f"If provided, limit_mode must be a string.\ntype(limit_mode): {type(limit_mode)}."
            )
        # limit_mode is a string.
        # Calculate limits appropriately.
        elif limit_mode == "stdev":
            # Verify stdevs.
            if not isinstance(stdevs, (int, float)):
                raise TypeError(
                    f"For limit_mode='stdev', <stdevs> must be of type int or float.\n"
                    f"type(stdevs): {type(stdevs)}."
                )
            if stdevs < 0:
                raise ValueError(
                    f"For limit_mode='stdev', <stdevs> must be non-negative.\n"
                    f"stdevs: {stdevs}."
                )
            # Choose limits equal to the mean +/- <stdevs> standard deviations.
            stdev = np.std(data)
            mean = np.mean(data)
            lower_lim = mean - stdevs * stdev
            upper_lim = mean + stdevs * stdev
        elif limit_mode == "quantile":
            # Verify quantile.
            if not isinstance(quantile, (int, float)):
                raise TypeError(
                    f"For limit_mode='quantile', <quantile> must be of type int or float.\n"
                    f"type(quantile): {type(quantile)}."
                )
            if quantile < 0 or quantile > 1:
                raise ValueError(
                    f"For limit_mode='quantile', <quantile> must be in the interval [0, 1].\n"
                    f"quantile: {quantile}."
                )
            # Choose limits based on quantile.
            lower_lim = np.quantile(data, min(quantile, 1 - quantile))
            upper_lim = np.quantile(data, max(quantile, 1 - quantile))
        else:
            raise ValueError(
                f"Unrecognized value for limit_mode. Supported values include {supported_limit_modes}.\n"
                f"limit_mode: {limit_mode}."
            )
    # lower_lim and upper_lim are set appropriately.

    # TODO: make harmonious with quantiles approach so that it centers at the median.
    # Scale data such that the bulk lies approximately on [0, 1].
    scaled_data = (data - lower_lim) / (upper_lim - lower_lim)

    # Return scaled copy of data.
    return scaled_data


def _validate_inputs(data, title, n_cuts, xcuts, ycuts, zcuts, figsize):
    """Returns a dictionary of the form {'inputName' : inputValue}.
    It has an entry for each argument that has been validated."""

    inputDict = {"data": data, "figsize": figsize}

    # Validate data.

    supported_data_types = [np.ndarray]
    # Convert data to np.ndarray for supported types.
    # if isinstance(data, ):
    #     # Convert to np.ndarray.
    #     data = np.array(data)

    # If data is none of the supported types, attempt to cast it as a np.ndarray.
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except TypeError:
            # If a TypeError was raised on casting data to a np.ndarray, raise informative TypeError.
            raise TypeError(
                f"data is not one of the supported types {supported_data_types} and cannot be cast as a np.ndarray.\n"
                f"type(data): {type(data)}."
            )

    if data.ndim < 3:  # <3
        raise ValueError(
            f"data must have at least 3 dimensions.\ndata.ndim: {data.ndim}."
        )
    # data is valid.
    inputDict.update(data=data)

    # Validate figsize.

    # TODO: dynamicize figsize.
    if figsize is None:
        # Compute figsize.
        raise NotImplementedError(
            "This functionality has not yet been implemented. Please provide another figsize."
        )
    else:
        try:
            if len(figsize) != 2:
                raise ValueError(
                    f"figsize must be an iterable with length 2.\n"
                    f"len(figsize): {len(figsize)}."
                )
        except TypeError:
            raise TypeError(
                f"figsize must be an iterable.\ntype(figsize): {type(figsize)}."
            )
    # figsize is valid.
    inputDict.update(figsize=figsize)

    return inputDict


# TODO: update work with interesting_cuts and new Image class
def _Image_to_Nifti2Image(image, affine=None):

    nifti2header = nib.Nifti2Header()
    nifti2header["dim"][1 : 1 + len(image.nxyz)] = image.nxyz
    nifti2header["pixdim"][1 : 1 + len(image.dxyz)] = image.dxyz

    nifti2image = nib.Nifti2Image(image.data, affine=affine, header=nifti2header)

    return nifti2image


def _get_cuts(data, xcuts, ycuts, zcuts, n_cuts=5, interesting_cuts=False):
    """Returns xcuts, ycuts, & zcuts. If any of these are provided, they are used.
    If any are not provided, they are computed.

    The default is to compute unprovided cuts as evenly spaced slices across that dimension.
    However, if interesting_cuts is True, then any dimension's cuts that are not specified
    are computed using niplot.find_cut_slices."""

    if interesting_cuts is True:
        # TODO: update Image_to_Nifti2Image to allow for multiple input types, and find a way to give it image metadata.
        raise NotImplementedError(
            "This functionality has not been fully implemented yet."
        )
        xcuts = (
            xcuts
            or niplot.find_cut_slices(
                Image_to_Nifti2Image(atlas_Image, affine=np.eye(4)),
                direction="x",
                n_cuts=n_cuts,
            ).astype(int)
        )
        ycuts = (
            ycuts
            or niplot.find_cut_slices(
                Image_to_Nifti2Image(atlas_Image, affine=np.eye(4)),
                direction="y",
                n_cuts=n_cuts,
            ).astype(int)
        )
        zcuts = (
            zcuts
            or niplot.find_cut_slices(
                Image_to_Nifti2Image(atlas_Image, affine=np.eye(4)),
                direction="z",
                n_cuts=n_cuts,
            ).astype(int)
        )
    else:
        xcuts = xcuts or np.linspace(0, data.shape[0], n_cuts + 2)[1:-1]
        ycuts = ycuts or np.linspace(0, data.shape[1], n_cuts + 2)[1:-1]
        zcuts = zcuts or np.linspace(0, data.shape[2], n_cuts + 2)[1:-1]

    return xcuts, ycuts, zcuts


# TODO: verify plotting works with xyzcuts provided with inconsistent lengths.
# TODO: allow n_cuts to be a triple.
def heatslices(
    data,
    title=None,
    figsize=(10, 5),
    cmap="gray",
    n_cuts=5,
    xcuts=[],
    ycuts=[],
    zcuts=[],
    limit_mode=None,
    stdevs=4,
    quantile=0.01,
    limits=None,
    vmin=0,
    vmax=1,
):
    """
    Produce a figure with 3 rows of images, each corresponding to a different orthogonal view of data.
    Each row can have arbitrarily many parallel views.
    The data is scaled such that its bulk lies on the interval [0, 1], with the extrema optionally left unaccounted for in determining the scaling.
    Those values outside the limits saturate at 0 or 1 in the figure.

    Args:
        data (np.ndarray): A 3 or 4 dimensional array containing volumetric intensity data (if 3D) or RGB data (if 4D) to be viewed.
        title (str, optional): The figure title. Defaults to None.
        figsize (tuple, optional): The width and height of the figure in inches. Defaults to (10, 5).
        cmap (str, optional): The name of the chosen color map. Defaults to 'gray'.
        n_cuts (int, optional): The number of parallel views to show on each row without a specified list of cuts. Defaults to 5.
        xcuts (list, optional): A list of indices at which to display a view in the first row. Defaults to [n_cuts evenly spaced indices. Half the spacing between indices pads each end].
        ycuts (list, optional): A list of indices at which to display a view in the second row. Defaults to [n_cuts evenly spaced indices. Half the spacing between indices pads each end].
        zcuts (list, optional): A list of indices at which to display a view in the third row. Defaults to [n_cuts evenly spaced indices. Half the spacing between indices pads each end.].
        limit_mode (str, NoneType, optional): A string indicating what mode to use for clipping the extrema of data for determining the scaling to the interval [vmin, vmax].
            Accepted values:
                - None
                - 'stdev'
                - 'quantile'
            Defaults to None.

        stdevs (float, optional): Used if limit_mode == 'stdev': The number of standard deviations from the mean that will be scaled to the interval [vmin, vmax]. Defaults to 4.
        quantile (float, optional): Used if limit_mode == 'quantile': The proportion of data that will not be considered for scaling to the interval [vmin, vmax]. Defaults to 0.01.
        limits (sequence, optional): The lower and upper limits bookmarking which values in data will be considered when scaling to the interval [vmin, vmax]. Overrides limit_mode. Defaults to None.
        vmin (float, optional): The smallest value displayed in the figure. Smaller values will saturate to vmin. Defaults to 0.
        vmax (float, optional): The largest value displayed in the figure. Larger values will saturate to vmax. Defaults to 1.
    """
    # Figure-tuning arguments.
    # What will be displayed.
    # data-scaling arguments.

    # TODO: validate all inputs.
    # Validate inputs
    inputs = {
        "data": data,
        "title": title,
        "n_cuts": n_cuts,
        "xcuts": xcuts,
        "ycuts": ycuts,
        "zcuts": zcuts,
        "figsize": figsize,
    }
    validated_inputs = _validate_inputs(**inputs)
    locals().update(validated_inputs)

    # Scale bulk of data to [0, 1].
    data = _scale_data(
        data, limit_mode=limit_mode, stdevs=stdevs, quantile=quantile, limits=limits
    )  # Side-effect: breaks alias.

    # Get cuts.
    xcuts, ycuts, zcuts = _get_cuts(data, xcuts, ycuts, zcuts, n_cuts)

    # maxcuts is the number of cuts in the dimension with the largest number of cuts.
    maxcuts = max(list(map(lambda cuts: len(cuts), [xcuts, ycuts, zcuts])))

    # TODO: check out imshow param extent for anisotropy
    # TODO: properly scale subplots / axs such that scale is consistent across all images.
    fig, axs = plt.subplots(3, maxcuts, sharex="row", sharey="row", figsize=figsize)
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.2
    )
    for ax in axs.ravel():
        pass
    plt.setp(axs, aspect="equal", xticks=[], yticks=[])
    for row, cuts in enumerate([xcuts, ycuts, zcuts]):
        for col, cut in enumerate(cuts):
            axs[row, col].grid(False)
            img = axs[row, col].imshow(
                data.take(cut, row), vmin=vmin, vmax=vmax, cmap=cmap
            )
    cax = plt.axes([0.925, 0.1, 0.025, 0.77])
    plt.colorbar(img, cax=cax)
    fig.suptitle(title, fontsize=20)
