import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


def plot_2d(img, title=None, margin=0.05, dpi=80, show_plot=True):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()

    if nda.ndim == 3:
        c = nda.shape[-1]

        if c not in (3, 4):
            nda = nda[nda.shape[0] // 2, :, :]

    elif nda.ndim == 4:
        c = nda.shape[-1]

        if c not in (3, 4):
            raise RuntimeError("Unable to show 3D-vector Image")

        nda = nda[nda.shape[0] // 2, :, :, :]

    xsize = nda.shape[1] * 2
    ysize = nda.shape[0] * 2

    figsize = (1 + margin) * xsize / dpi, (1 + margin) * ysize / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    if show_plot:
        extent = (0, xsize * spacing[0], ysize * spacing[1], 0)

        t = ax.imshow(nda, extent=extent, interpolation=None)

        if nda.ndim == 2:
            t.set_cmap("gray")

        if title:
            plt.title(title)

        plt.show()

    return fig, ax


def plot_3d(
    img,
    xslices=[],
    yslices=[],
    zslices=[],
    title=None,
    margin=0.05,
    dpi=80,
    show_plot=True,
):
    if not isinstance(img, sitk.SimpleITK.Image):
        raise Exception("Sorry, input must be an sitk image")
    else:

        img_xslices = [img[s, :, :] for s in xslices]
        img_yslices = [img[:, s, :] for s in yslices]
        img_zslices = [img[:, :, s] for s in zslices]

        maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))

        img_null = sitk.Image(
            [0, 0], img.GetPixelID(), img.GetNumberOfComponentsPerPixel()
        )

        img_slices = []
        d = 0

        if len(img_xslices):
            img_slices += img_xslices + [img_null] * (maxlen - len(img_xslices))
            d += 1

        if len(img_yslices):
            img_slices += img_yslices + [img_null] * (maxlen - len(img_yslices))
            d += 1

        if len(img_zslices):
            img_slices += img_zslices + [img_null] * (maxlen - len(img_zslices))
            d += 1

        if maxlen != 0:
            if img.GetNumberOfComponentsPerPixel() == 1:
                img = sitk.Tile(img_slices, [maxlen, d])
            else:
                img_comps = []
                for i in range(0, img.GetNumberOfComponentsPerPixel()):
                    img_slices_c = [
                        sitk.VectorIndexSelectionCast(s, i) for s in img_slices
                    ]
                    img_comps.append(sitk.Tile(img_slices_c, [maxlen, d]))
                img = sitk.Compose(img_comps)

    return plot_2d(img, title, margin, dpi, show_plot)


def plot_image_2d(imgs, titles=None, rows=1, colorbar=False):
    """
    Plots a 2D image

    Parameters
    ----------
    imgs : single 2D np.array or list of 2D np.arrays
        slice of image data frog ngl_pipeline class.
        example use: img_slice = img[:,100,:]

    titles : str, default = None

    Returns
    -------
    matplotlib figure and axis
    """
    if isinstance(imgs, list):
        l = len(imgs)
        cols = np.ceil(l / rows).astype(int)
        fig, axes = plt.subplots(rows, cols)

        for i, img in enumerate(imgs):
            img = np.swapaxes(img, 0, 1)
            ax = axes[i]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if titles is not None:
                ax.set_title(titles[i])
        return fig, axes
    else:
        img = np.swapaxes(imgs, 0, 1)
        fig, axis = plt.subplots()
        im = axis.imshow(img)
        axis.set_xticks([])
        axis.set_yticks([])
        if colorbar:
            plt.colorbar(im, ax=axis)
        return fig, axis


def plot_image_mip(imgs, titles=None, rows=1, axis=2, colorbar=False):
    """
    Max Intensity Projection of 3D image

    Parameters
    ----------
    imgs : single 3D np.array or list of 3D np.arrays

    titles : str, default = None

    Returns
    -------
    matplotlib figure and axis
    """
    if isinstance(imgs, list):
        for i, img in enumerate(imgs):
            imgs[i] = np.amax(img, axis)
        fig, axes = plot_image_2d(imgs, rows=rows, titles=titles, colorbar=colorbar)
        return fig, axes
    else:
        mip = np.amax(imgs, axis)
        fig, axes = plot_image_2d(mip, titles=titles, colorbar=colorbar)
        return fig, axes


def find_smalldim(imgs):
    """
    Find smallest dimension of an image or list of images.

    Parameters
    ----------
    imgs : single 3D np.array or list of 3D np.arrays {

    titles : str, default = None

    Returns
    -------
    smallest_axis: int
    """
    if isinstance(imgs, list):
        img = imgs[0]
    else:
        img = imgs
    shp = img.shape
    smallest_axis = np.argmin(shp)
    return int(smallest_axis)


def plot_image_hist(imgs, rows=1, titles=None):
    """
    Histogram

    Parameters
    ----------
    imgs : single 3D np.array or list of 3D np.arrays {

    titles : str, default = None

    Returns
    -------
    matplotlib figure and axis
    """
    if isinstance(imgs, list):
        l = len(imgs)
        cols = np.ceil(l / rows).astype(int)
        fig, axes = plt.subplots(rows, cols)

        for i, img in enumerate(imgs):
            img = np.swapaxes(img, 0, 1)
            ax = axes[i]
            ax.hist(img.flatten(), bins=50)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_xlabel("Intensity", fontsize=12)
            if titles is not None:
                ax.title(titles[i], fontsize=12)
        return fig, axes
    else:
        histo = plt.hist(imgs.flatten())
        plt.ylabel("Count", fontsize=12)
        plt.xlabel("Intensity", fontsize=12)
        return histo


def plot_image_pts(img, voxels, colorbar=False):
    """
    Visualize a chunk around a voxel

    Parameters
    ----------
    img : single 2D np.array

    voxels : np.array
        voxel output from ngl_pipeline class

    Returns
    -------
    matplotlib figure and axis
    """
    fig, axes = plt.subplots()

    if len(voxels.shape) > 1:
        im = axes.imshow(img)
        axes.scatter(voxels[:, 0], voxels[:, 1])
    else:
        im = axes.imshow(img)
        axes.scatter(voxels[0], voxels[1])

    if colorbar:
        plt.colorbar(im, ax=axes)

    return fig, axes
