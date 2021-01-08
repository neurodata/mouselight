import numpy as np
from scipy.special import factorial
from functools import reduce
from itertools import product

from ..lddmm._lddmm_utilities import _validate_ndarray
from ..lddmm._lddmm_utilities import _validate_scalar_to_multi
from ..lddmm._lddmm_utilities import _compute_coords
from ..lddmm._lddmm_utilities import _validate_resolution


def _generate_rotation_matrix(angles):

    raise NotImplementedError(
        "This function, _generate_rotation_matrix, has not been completed."
    )

    angles = _validate_ndarray(angles, required_ndim=1)

    ndim = len(
        angles
    )  # TODO: this is wrong, for ndim > 3, len(angles) > ndim. ndim_choose_2 = len(angles).

    n_rotation_planes = factorial(ndim) / (factorial(ndim - 2) * factorial(2))
    n_rotation_planes = ndim * (ndim - 1)

    rotation_matrix = np.zeros((n_rotation_planes, ndim, ndim))

    for indices in product(*map(range, rotation_matrix.shape)):

        if min(indices) == max(indices):
            rotation_matrix[indices] = 1

        if True:
            pass

    for rotation_plane_index in range(n_rotation_planes):
        # Rotation planes are ordered such that they are cyclically fixed.
        rotation_plane = rotation_plane_index % ndim, None
        for indices in product(*map(range, rotation_matrix.shape[1:])):
            # TODO: verify order.
            corner_index = rotation_plane // None

    rotation_matrix = reduce(np.matmul, rotation_matrix)

    return rotation_matrix


def locally_rotate_velocity_fields(
    velocity_fields,
    resolution,
    blob_center,
    blob_width,
    rotation_angles,
    rotation_center,
):
    """
    Mutate velocity_fields by applying a rotation continuously to a local region centered on a gaussian blob that weights the displacements from rotating.

    Args:
        velocity_fields (np.ndarray): The velocity_fields to be mutated. Should have shape equal to (*image.shape, num_timesteps, image.ndim).
        resolution (float, seq): The resolution of the image underlying velocity_fields.
        blob_center (float, seq): The coordinates of the rotation center in real units.
        blob_width (float, seq): The standard deviation of the blob per spatial dimension determining where the rotation affects.
        rotation_angles (float, seq): The rotation angles per spatial dimension in units of degrees.
        rotation_center (float, seq): The coordinates of the point about which the rotation occurs in real units.
    """

    # Note: mutates velocity_fields.

    # Note: only implemented for 3D.

    # Validate inputs.
    velocity_fields = _validate_ndarray(velocity_fields, required_ndim=3 + 2)
    blob_center = _validate_scalar_to_multi(blob_center, velocity_fields.ndim - 2)
    blob_width = _validate_scalar_to_multi(blob_width, velocity_fields.ndim - 2)
    rotation_angles = (
        _validate_scalar_to_multi(rotation_angles, velocity_fields.ndim - 2)
        * np.pi
        / 180
    )
    rotation_center = _validate_scalar_to_multi(
        rotation_center, velocity_fields.ndim - 2
    )

    # Construct and apply rotations.

    spatial_ndim = velocity_fields.ndim - 2
    identity_position_field = _compute_coords(
        velocity_fields.shape[:spatial_ndim], resolution
    )
    blob_weights = np.exp(
        -np.sum(
            (identity_position_field - blob_center) ** 2 / (2 * blob_width ** 2),
            axis=-1,
        )
    )

    # Note: to make this function n-dimensional, define the rotation_matrix dynamically.
    rotation_matrix = np.array(
        [
            [
                [1, 0, 0],
                [0, np.cos(rotation_angles[1]), -np.sin(rotation_angles[1])],
                [0, np.sin(rotation_angles[1]), np.cos(rotation_angles[1])],
            ],
            [
                [np.cos(rotation_angles[2]), 0, np.sin(rotation_angles[2])],
                [0, 1, 0],
                [-np.sin(rotation_angles[2]), 0, np.cos(rotation_angles[2])],
            ],
            [
                [np.cos(rotation_angles[0]), -np.sin(rotation_angles[0]), 0],
                [np.sin(rotation_angles[0]), np.cos(rotation_angles[0]), 0],
                [0, 0, 1],
            ],
        ]
    )
    rotation_matrix = reduce(np.matmul, reversed(rotation_matrix))

    # rotation_displacements = (rotation_matrix @ ((identity_position_field - rotation_center).reshape(-1, spatial_ndim).T)).T.reshape(identity_position_field.shape) - (identity_position_field - rotation_center)
    rotation_displacements = (
        (
            rotation_matrix
            @ ((identity_position_field - rotation_center).reshape(-1, spatial_ndim).T)
        ).T.reshape(identity_position_field.shape)
    ) - (identity_position_field - rotation_center)

    velocity_fields += (
        np.expand_dims(rotation_displacements * blob_weights[..., None], -2)
        / velocity_fields.shape[-2]
    )


def locally_translate_velocity_fields(
    velocity_fields, resolution, blob_center, blob_width, translations
):
    """
    Mutate velocity_fields by applying a translation continuously to a local region centered on a gaussian blob that weights the displacements from translating.

    Args:
        velocity_fields (np.ndarray): The velocity_fields to be mutated. Should have shape equal to (*image.shape, num_timesteps, image.ndim).
        resolution (float, seq): The resolution of the image underlying velocity_fields.
        blob_center (float, seq): The coordinates of the rotation center in real units.
        blob_width (float, seq): The standard deviation of the blob per spatial dimension determining where the rotation affects.
        translations (float, seq): The translations per spatial dimension in real units.
    """

    # Note: mutates velocity_fields.

    # Validate inputs.
    velocity_fields = _validate_ndarray(velocity_fields, minimum_ndim=2)
    resolution = _validate_resolution(resolution, velocity_fields.ndim - 2)
    blob_center = _validate_scalar_to_multi(blob_center, velocity_fields.ndim - 2)
    blob_width = _validate_scalar_to_multi(blob_width, velocity_fields.ndim - 2)
    translations = _validate_scalar_to_multi(translations, velocity_fields.ndim - 2)

    spatial_ndim = velocity_fields.ndim - 2
    identity_position_field = _compute_coords(
        velocity_fields.shape[:spatial_ndim], resolution
    )
    blob_weights = np.exp(
        -np.sum(
            (identity_position_field - blob_center) ** 2 / (2 * blob_width ** 2),
            axis=-1,
        )
    )

    translation_displacements = blob_weights[..., None] * translations

    velocity_fields += (
        np.expand_dims(translation_displacements, -2) / velocity_fields.shape[-2]
    )
