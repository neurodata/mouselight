import numpy as np


class ImageMetaData:
    """Container class for metadata about an image. Stores:
    - nxyz (np.ndarray): image shape - computed from provided image.
    - dxyz (np.ndarray or scalar): image resolution - if dxyz is a scalar, it is upcast to the length of nxyz.
    - xyz (np.ndarray): image coordinates - computed, not provided.
    - origin (np.ndarray): origin for xyz coordinates - default is center.
        Options:
        - 'center'
        - 'zero'
        - np.ndarray of the same length as nxyz
        - scalar, upcast to np.ndarray of same length as nxyz"""

    def __init__(self, dxyz, nxyz=None, image=None, origin="center", name=None):
        """If nxyz is provided, use it as nxyz, the image's shape.
        If image is provided, use its shape as nxyz.
        If both nxyz and image are provided and indicate different shapes, raise error.
        If neither nxyz nor image are provided, raise error.
        If origin is provided, it is used to compute xyz.
        If origin is not provided, xyz is centered by default."""

        # Instantiate attributes.
        self.dxyz = None
        self.nxyz = None
        self.xyz = None
        self.name = None

        # Populate attributes.

        # nxyz attribute.
        self.nxyz = ImageMetaData._validate_nxyz(nxyz, image)

        # dxyz attribute.
        self.dxyz = ImageMetaData._validate_dxyz(dxyz, self.nxyz)

        # xyz attribute.
        self.xyz = ImageMetaData._generate_xyz(dxyz, nxyz, origin)

        # name attribute.
        self.name = name

    @staticmethod
    def _validate_nxyz(nxyz, image) -> np.ndarray:
        """Validate compatibility between nxyz and image as provided,
        and return an appropriate value for the nxyz attribute."""

        # Validate agreement between nxyz and image.
        if nxyz is not None and image is not None:
            if not all(nxyz == np.array(image).shape):
                raise ValueError(
                    f"nxyz and image were both provided, but nxyz does not match the shape of image."
                    f"\nnxyz: {nxyz}, image.shape: {image.shape}."
                )

        # If nxyz is provided but image is not.
        if nxyz is not None:
            # Cast as np.ndarray.
            if not isinstance(nxyz, np.ndarray):
                nxyz = np.array(nxyz)  # Side effect: breaks alias.
            # If nxyz is multidimensional, raise error.
            if nxyz.ndim > 1:
                raise ValueError(
                    f"nxyz cannot be multidimensional.\nnxyz.ndim: {nxyz.ndim}."
                )
            # If nxyz is 0-dimensional, upcast to 1-dimensional. A perverse case.
            if nxyz.ndim == 0:
                nxyz = np.array([nxyz])
            # If nxyz is 1-dimensional, set nxyz attribute.
            if nxyz.ndim == 1:
                return nxyz
        # If image is provided but nxyz is not.
        # Will fail if image cannot be cast as a np.ndarray.
        elif image is not None:
            # Cast as np.ndarray.
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            # If image is 0-dimensional, upcast to 1-dimensional. A perverse case.
            if image.ndim == 0:
                image = np.array([image])
            # image is a non-zero-dimensional np.ndarray. Set nxyz attribute.
            return np.array(image.shape)
        else:
            raise RuntimeError(
                f"At least one of nxyz and image must be provided. Both were received as their default value: None."
            )

    @staticmethod
    def _validate_dxyz(dxyz, nxyz: np.ndarray) -> np.ndarray:
        """Validate dxyz and its compatibility with nxyz.
        Return an appropriate value for the dxyz attribute."""

        # Cast as np.ndarray.
        if not isinstance(dxyz, np.ndarray):
            dxyz = np.array(dxyz)  # Side effect: breaks alias.
        # if dxyz is multidimensional, raise error.
        if dxyz.ndim > 1:
            raise ValueError(
                f"dyxz must be 1-dimensional. The value provided is {dxyz.ndim}-dimensional."
            )
        # If dxyz is 0-dimensional, upcast to match the length of nxyz.
        if dxyz.ndim == 0:
            dxyz = np.array([dxyz] * len(nxyz))
        # dxyz is 1-dimensional.
        if len(dxyz) == len(nxyz):
            # dxyz is 1-dimensional and matches the length of nxyz. Set dxyz attribute.
            return dxyz
        else:
            raise ValueError(
                f"dyxz must be either 0-dimensional or 1-dimensional and match the length of nxyz or the shape of image."
                f"\nlen(dxyz): {len(dxyz)}."
            )

    @staticmethod
    def _generate_xyz(dxyz: np.ndarray, nxyz: np.ndarray, origin) -> np.ndarray:
        """Generate image coordinates xyz based on the resolution <dxyz>, the shape <nxyz>, and the center designation <origin>."""

        # Instantiate xyz.
        # xyz is a list of np.ndarray objects of type float and represents the physical coordinates in each dimension.
        xyz = [
            np.arange(nxyz_i).astype(float) * dxyz_i
            for nxyz_i, dxyz_i in zip(nxyz, dxyz)
        ]

        # Shift xyz.
        if isinstance(origin, str):
            if origin == "center":
                # Offset by the mean along each dimension.
                for coords in xyz:
                    coords -= np.mean(coords)
            elif origin == "zero":
                # This designation is how xyz was initially created and thus it requires no shift.
                pass
            else:
                raise ValueError(
                    f"Unsupported value for origin. Supported string values include ['center', 'zero']."
                    f"\norigin: {origin}."
                )
        elif isinstance(origin, (int, float, list, np.ndarray)):
            if isinstance(origin, (int, float, list)):
                # Cast to np.ndarray.
                origin = np.array(origin)
            # origin is a np.ndarray.
            # If origin has length 1, broadcast to match nxyz.
            if origin.ndim == 0:
                origin = np.array(list(origin) * len(nxyz))
            # If the length of origin matches the length of nxyz, perform offset in each dimension.
            if len(origin) == len(nxyz):
                for dim, coords in enumerate(xyz):
                    coords -= origin[dim]
            else:
                raise ValueError(
                    f"origin must either be a scalar, have length 1, or match the length of nxyz or the shape of image."
                    f"\nlen(origin): {len(origin)}."
                )
        else:
            raise ValueError(
                f"must be one of the following types: [str, int, float, list, np.ndarray]."
                f"\ntype(origin): {type(origin)}."
            )

        # xyz has been created from dxyz and nxyz, and then adjusted appropriately based on origin.
        return xyz


class Image(ImageMetaData):
    """Subclass of ImageMetaData that also stores the image data itself.
    Attributes:
        - image, the image data
        - dxyz, the resolution in each dimension
        - nxyz, the shape of the image
        - xyz, the coordinates of the image
        - name, optional name"""

    def __init__(self, image, dxyz, origin="center", name=None):
        """Validation is handled by ImageMetaData.__init__."""

        super().__init__(dxyz=dxyz, image=image, origin=origin, name=name)

        # Redo validation on image to set image attribute.
        image = np.array(image)  # Side effect: breaks alias
        # If image is 0-dimensional, upcast to 1-dimensional. A perverse case.
        if image.ndim == 0:
            image = np.array([image])
        # Set image attribute.
        self.image = image
