# Transform is the primary class in ARDENT.

import numpy as np
from pathlib import Path
import pickle

from .presets import get_registration_preset
from .lddmm._lddmm import lddmm_register, lddmm_transform_image, lddmm_transform_points
from . import file_io


class Transform:
    """
    Transform stores the deformation that is output by a registration
    and provides methods for applying that transformation to various images.
    """

    def __init__(self):
        """
        Initialize Transform object. Sets attributes to None.
        """

        # Create attributes.

        # Saved for the continue_registration method.
        self._registration_parameters = None

        # lddmm_dict.

        # Core.
        self.affine = (None,)
        self.phi = (None,)
        self.phi_inv = (None,)
        self.affine_phi = (None,)
        self.phi_inv_affine_inv = (None,)
        self.contrast_coefficients = (None,)
        self.velocity_fields = None

        # Helpers.
        self.template_resolution = (None,)
        self.target_resolution = (None,)

        # Accumulators.
        self.matching_energies = (None,)
        self.regularization_energies = (None,)
        self.total_energies = (None,)

        # Debuggers.
        self.lddmm = (None,)

    def _update_lddmm_attributes(self, lddmm_dict):
        """Update attributes with the output dictionary from lddmm_register."""

        # Verify lddmm_dict.
        if not isinstance(lddmm_dict, dict):
            raise TypeError(
                f"lddmm_dict must be of type dict.\n"
                f"type(lddmm_dict): {type(lddmm_dict)}."
            )

        # Set attributes.

        for key, value in lddmm_dict.items():
            if not hasattr(self, key):
                raise ValueError(
                    f"lddmm_dict must only have keys that are attributes of self.\n"
                    f"key: {key}."
                )
            setattr(self, key, value)

    def get_lddmm_dict(self):
        """
        Constructs lddmm_dict, a dictionary of this object's attributes as populated by the return from lddmm_register.

        Returns:
            dict: The attributes of this Transform object, matching the dictionary returned from lddmm_register.
        """

        return dict(
            # Core.
            affine=self.affine,
            phi=self.phi,
            phi_inv=self.phi_inv,
            affine_phi=self.affine_phi,
            phi_inv_affine_inv=self.phi_inv_affine_inv,
            contrast_coefficients=self.contrast_coefficients,
            velocity_fields=self.velocity_fields,
            # Helpers.
            template_resolution=self.template_resolution,
            target_resolution=self.target_resolution,
            # Accumulators.
            matching_energies=self.matching_energies,
            regularization_energies=self.regularization_energies,
            total_energies=self.total_energies,
            # Debuggers.
            lddmm=self.lddmm,
        )

    def register(
        self,
        # Images.
        template,
        target,
        # Image resolutions.
        template_resolution=None,
        target_resolution=None,
        # Multiscale.
        multiscales=None,
        # Preset.
        preset=None,
        # Iterations.
        num_iterations=None,
        num_affine_only_iterations=None,
        num_rigid_affine_iterations=None,
        # Stepsizes.
        affine_stepsize=None,
        deformative_stepsize=None,
        # Affine specifiers.
        fixed_affine_scale=None,
        # Velocity field specifiers.
        sigma_regularization=None,
        velocity_smooth_length=None,
        preconditioner_velocity_smooth_length=None,
        maximum_velocity_fields_update=None,
        num_timesteps=None,
        # Contrast map specifiers.
        contrast_order=None,
        spatially_varying_contrast_map=None,
        contrast_iterations=None,
        sigma_contrast=None,
        contrast_smooth_length=None,
        # Smoothness vs. accuracy tradeoff.
        sigma_matching=None,
        # Classification specifiers.
        classify_and_weight_voxels=None,
        sigma_artifact=None,
        sigma_background=None,
        artifact_prior=None,
        background_prior=None,
        # Initial values.
        initial_affine=None,
        initial_velocity_fields=None,
        initial_contrast_coefficients=None,
        # Diagnostic outputs.
        calibrate=None,
        track_progress_every_n=None,
    ):
        """
        Compute a registration between template and target, to be applied with lddmm_transform_image.

        Args:
            template (np.ndarray): The ideally clean template image being registered to the target.
            target (np.ndarray): The potentially messier target image being registered to.
            template_resolution (float, list, optional): A scalar or list of scalars indicating the resolution of the template. Overrides 0 input. Defaults to 1.
            target_resolution (float, optional): A scalar or list of scalars indicating the resolution of the target. Overrides 0 input. Defaults to 1.
            multiscales (float, seq, optional): A scalar, list of scalars, or list of lists or np.ndarray of scalars, determining the levels of downsampling at which the registration should be performed before moving on to the next.
                Values must be either all at least 1, or all at most 1. Both options are interpreted as downsampling. For example, multiscales=[10, 3, 1] will result in the template and target being downsampled by a factor of 10 and registered.
                This registration will be upsampled and used to initialize another registration of the template and target downsampled by 3, and then again on the undownsampled data. multiscales=[1/10, 1/3, 1] is equivalent.
                Alternatively, the scale for each dimension can be specified, e.g. multiscales=[ [10, 5, 5], [3, 3, 3], 1] for a 3D registration will result in the template and target downsampled by [10, 5, 5], then [3, 3, 3], then [1, 1, 1].
                If provided with more than 1 value, all following arguments with the exceptions of initial_affine, initial_velocity_fields, and initial_contrast_coefficients,
                which may be provided for the first value in multiscales, may optionally be provided as sequences with length equal to the number of values provided to multiscales. Each such value is used at the corresponding scale.
                Additionally, template_resolution and target_resolution cannot be provided for each scale in multiscales. Rather, they are given once to indicate the resolution of the template and target as input.
                multiscales should be provided as descending values. Defaults to 1.
            preset (str, optional): A string specifying a recognized preset, or subset of the following arguments to provide automatically, overridden by values specified in this call. Defaults to None.
            num_iterations (int, optional): The total number of iterations. Defaults to 300.
            num_affine_only_iterations (int, optional): The number of iterations at the start of the process without deformative adjustments. Defaults to 100.
            num_rigid_affine_iterations (int, optional): The number of iterations at the start of the process in which the affine is kept rigid. Defaults to 50.
            affine_stepsize (float, optional): The unitless stepsize for affine adjustments. Should be between 0 and 1. Defaults to 0.3.
            deformative_stepsize (float, optional): The stepsize for deformative adjustments. Optimal values are problem-specific. Setting preconditioner_velocity_smooth_length increases the appropriate value of deformative_stepsize.
                If equal to 0 then the result is affine-only registration. By default 0.
            fixed_affine_scale (float, optional): The scale to impose on the affine at all iterations. If None, no scale is imposed. Otherwise, this has the effect of making the affine always rigid. Defaults to None.
            sigma_regularization (float, optional): A scalar indicating the freedom to deform. Small values put harsher constraints on the smoothness of a deformation.
                With sufficiently large values, the registration will overfit any noise in the target, leading to unrealistic deformations.
                However, this may still be appropriate with a small num_iterations. Overrides 0 input. Defaults to np.inf.
            velocity_smooth_length (float, optional): The length scale of smoothing. Overrides 0 input. Defaults to 2 * np.max(self.template_resolution).
            preconditioner_velocity_smooth_length (float, optional): The length of preconditioner smoothing of the velocity_fields in physical units. Affects the optimization of the velocity_fields, but not the optimum. Defaults to 0.
            maximum_velocity_fields_update (float, optional): The maximum allowed update to the velocity_fields in physical units. Affects the optimization of the velocity_fields, but not the optimum.
                Overrides 0 input. Defaults to np.max(self.template.shape * self.template_resolution).
            num_timesteps (int, optional): The number of composed sub-transformations in the diffeomorphism. Overrides 0 input. Defaults to 5.
            contrast_order (int, optional): The order of the polynomial fit between the contrasts of the template and target. Overrides 0 input. Defaults to 1.
            spatially_varying_contrast_map (bool, optional): If True, uses a polynomial per voxel to compute the contrast map rather than a single polynomial. Defaults to False.
            contrast_iterations (int, optional): The number of iterations of gradient descent to converge toward the optimal contrast_coefficients if spatially_varying_contrast_map == True. Overrides 0 input. Defaults to 5.
            sigma_contrast (float, optional): The scale of variation in the contrast_coefficients if spatially_varying_contrast_map == True. Overrides 0 input. Defaults to 1e-2.
            contrast_smooth_length (float, optional): The length scale of smoothing of the contrast_coefficients if spatially_varying_contrast_map == True. Overrides 0 input. Defaults to 2 * np.max(self.target_resolution).
            sigma_matching (float, optional): An estimate of the spread of the noise in the target,
                representing the tradeoff between the regularity and accuracy of the registration, where a smaller value should result in a less smooth, more accurate result.
                Typically it should be set to an estimate of the standard deviation of the noise in the image, particularly with artifacts. Overrides 0 input. Defaults to the standard deviation of the target.
            classify_and_weight_voxels (bool, optional): If True, artifacts are jointly classified with registration using sigma_artifact. Defaults to False.
            sigma_artifact (float, optional): The level of expected variation between artifact and non-artifact intensities. Overrides 0 input. Defaults to 5 * sigma_matching.
            sigma_background (float, optional): The level of expected variation between background and non-background intensities. Overrides 0 input. Defaults to 2 * sigma_matching.
            artifact_prior (float, optional): The prior probability at which we expect to find that any given voxel is artifact. Defaults to 1/3.
            background_prior (float, optional): The prior probability at which we expect to find that any given voxel is background. Defaults to 1/3.
            initial_affine (np.ndarray, optional): The affine array that the registration will begin with. Defaults to np.eye(template.ndim + 1).
            initial_velocity_fields (np.ndarray, optional): The velocity fields that the registration will begin with. Defaults to all zeros.
            initial_contrast_coefficients (np.ndarray, optional): The contrast coefficients that the registration will begin with.
                If None, the 0th order coefficient(s) are set to np.mean(self.target) - np.mean(self.template) * np.std(self.target) / np.std(self.template),
                if self.contrast_order > 1, the 1st order coefficient(s) are set to np.std(self.target) / np.std(self.template),
                and all others are set to zero. Defaults to None.
            calibrate (bool, optional): A boolean flag indicating whether to accumulate additional intermediate values and display informative plots for calibration purposes. Defaults to False.
            track_progress_every_n (int, optional): If positive, a progress update will be printed every track_progress_every_n iterations of registration. Defaults to 0.
        """

        # Collect registration parameters.
        registration_parameters = dict(
            # Images.
            template=template,
            target=target,
            # Image resolutions.
            template_resolution=template_resolution,
            target_resolution=target_resolution,
            # Multiscale.
            multiscales=multiscales,
            # Iterations.
            num_iterations=num_iterations,
            num_affine_only_iterations=num_affine_only_iterations,
            num_rigid_affine_iterations=num_rigid_affine_iterations,
            # Stepsizes.
            affine_stepsize=affine_stepsize,
            deformative_stepsize=deformative_stepsize,
            # Affine specifiers.
            fixed_affine_scale=fixed_affine_scale,
            # Velocity field specifiers.
            sigma_regularization=sigma_regularization,
            velocity_smooth_length=velocity_smooth_length,
            preconditioner_velocity_smooth_length=preconditioner_velocity_smooth_length,
            maximum_velocity_fields_update=maximum_velocity_fields_update,
            num_timesteps=num_timesteps,
            # Contrast map specifiers.
            contrast_order=contrast_order,
            spatially_varying_contrast_map=spatially_varying_contrast_map,
            contrast_iterations=contrast_iterations,
            sigma_contrast=sigma_contrast,
            contrast_smooth_length=contrast_smooth_length,
            # # vs. accuracy tradeoff.
            sigma_matching=sigma_matching,
            # Classification specifiers.
            classify_and_weight_voxels=classify_and_weight_voxels,
            sigma_artifact=sigma_artifact,
            sigma_background=sigma_background,
            artifact_prior=artifact_prior,
            background_prior=background_prior,
            # Initial values.
            initial_affine=initial_affine,
            initial_velocity_fields=initial_velocity_fields,
            initial_contrast_coefficients=initial_contrast_coefficients,
            # Diagnostic outputs.
            calibrate=calibrate,
            track_progress_every_n=track_progress_every_n,
        )

        # Fill unspecified parameters with presets if applicable.
        if preset is not None:
            preset_registration_parameters = get_registration_preset(preset)
            for registration_parameter in preset_registration_parameters.keys():
                # Override a registration_parameter with a preset value only if that parameter was not specified in this call (i.e. is None).
                if registration_parameters[registration_parameter] is None:
                    registration_parameters[
                        registration_parameter
                    ] = preset_registration_parameters[registration_parameter]

        # Perform registration.
        lddmm_dict = lddmm_register(**registration_parameters)

        # Save registration parameters for the continue_registration method, with the initial_affine, initial_velocity_fields, and initial_contrast_coefficients updated.
        registration_parameters.update(
            initial_affine=lddmm_dict["affine"],
            initial_contrast_coefficients=lddmm_dict["contrast_coefficients"],
            initial_velocity_fields=lddmm_dict["velocity_fields"],
        )
        self._registration_parameters = registration_parameters

        # Update attributes.
        self._update_lddmm_attributes(lddmm_dict)

    def continue_registration(self, **registration_parameter_updates):
        """
        Continue registering with all the same registration parameters from the previous call to the register method,
        but with initial_affine, initial_velocity_fields, and initial_contrast_coefficients set to the affine, velocity_fields, and contrast_coefficients
        most recently calculated in the register method, updated by registration_parameter_updates.

        Kwargs:
            registration_parameter_updates (key-value pairs, optional): registration parameters provided as kwargs
                to overwrite the most recent registration_parameters used in the register method. Defaults to {}.

        Raises:
            RuntimeError: Raised if self._registration_parameters are not set (as they should be after a call to the register method).
        """

        if self._registration_parameters is None:
            raise RuntimeError(
                f"The continue_registration method cannot be called from an object that has not performed the register method first."
            )

        # Update most recently saved registration parameters with user-provided kwargs.
        self._registration_parameters.update(registration_parameter_updates)

        # Continue the registration.
        self.register(**self._registration_parameters)

    def transform_image(
        self,
        subject,
        subject_resolution=1,
        output_resolution=None,
        output_shape=None,
        deform_to="template",
        extrapolation_fill_value=None,
        save_path=None,
    ):
        """
        Apply the transformation--computed by the last call to self.register--to subject,
        deforming it into the space of deform_to.

        Args:
            subject (np.ndarray): The image to deform.
            subject_resolution (float, seq, optional): The resolution of subject in each dimension, or just one scalar to indicate isotropy. Defaults to 1.
            deform_to (str, optional): Either 'template' or 'target' indicating which to deform subject to match. Defaults to: "template".
            output_resolution (float, seq, optional): The resolution of the output deformed_subject in each dimension,
                or just one scalar to indicate isotropy, or None to indicate the resolution of template or target based on deform_to.
                Cannot be provided along with output_resolution. Defaults to None.
            output_shape (seq, optional): The shape of the output deformed_subject, or None to indicate the shape of the template or target based on deform_to.
                Cannot be provided along with output_resolution. By default None.
            extrapolation_fill_value (float, NoneType, optional): The fill_value kwarg passed to scipy.interpolate.interpn; it should be background intensity.
                If None, this is set to a low quantile of the subject's 10**-subject.ndim quantile to estimate background. Defaults to None.
            save_path (str, Path, optional): The full path to save the output to. Defaults to: None.

        Returns:
            np.ndarray: The result of deforming subject to match deform_to.
        """

        transformed_subject = lddmm_transform_image(
            subject=subject,
            subject_resolution=subject_resolution,
            output_resolution=output_resolution,
            output_shape=output_shape,
            deform_to=deform_to,
            extrapolation_fill_value=extrapolation_fill_value,
            **self.get_lddmm_dict(),
        )

        if save_path is not None:
            file_io.save(transformed_subject, save_path)

        return transformed_subject

    def transform_points(self, points, deform_to="template", save_path=None):
        """
        Apply the transformation--computed by the last call to self.register--to points,
        deforming them into the space of deform_to.

        Args:
            points (np.ndarray): The points to deform. The last dimension should contain the coordinates for each point.
            deform_to (str, optional): Either 'template' or 'target' indicating which to deform points to match. Defaults to: "template".
            save_path (str, Path, optional): The full path to save the output to. Defaults to: None.

        Returns:
            np.ndarray: The result of deforming points to match deform_to, in the same shape as points was given.
        """

        transformed_points = lddmm_transform_points(
            points=points,
            deform_to=deform_to,
            affine_phi=self.affine_phi,
            phi_inv_affine_inv=self.phi_inv_affine_inv,
            template_resolution=self.template_resolution,
            target_resolution=self.target_resolution,
        )

        if save_path is not None:
            file_io.save(transformed_points, save_path)

        return transformed_points

    def save(self, file_path):
        """
        Save the entire instance of this Transform object (self) to file.

        Args:
            file_path (str, Path): The full path to save self to.
        """

        file_io.save_pickled(self, file_path)

    def load(self, file_path):
        """
        Load an entire instance of a Transform object from memory, as from a file created with the save method,
        and transplants all of its writeable attributes into self.

        Args:
            file_path (str, Path): The full path that a Transform object was saved to.
        """

        transform = file_io.load_pickled(file_path)

        self.__dict__.update(transform.__dict__)
