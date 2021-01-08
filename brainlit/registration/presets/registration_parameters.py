preset_parameters = {}

# Define registration parameter presets.
preset_parameters.update(
    {
        "identity": dict(
            translational_stepsize=0,
            linear_stepsize=0,
            deformative_stepsize=0,
            sigma_regularization=0,
            num_affine_only_iterations=0,
            num_iterations=1,
        )
    }
)
preset_parameters.update(
    {
        "clarity, mouse": dict(
            translational_stepsize=1e-5,
            linear_stepsize=5e-8,
            deformative_stepsize=5e-1,
            sigma_regularization=2e1,
        )
    }
)
preset_parameters.update(
    {
        "nissl, mouse": dict(
            translational_stepsize=2e-9,
            linear_stepsize=1e-13,
            deformative_stepsize=5e-4,
            sigma_regularization=1e0,
        )
    }
)
preset_parameters.update(
    {
        "mri, human": dict(
            translational_stepsize=1e-9,
            linear_stepsize=5e-13,
            deformative_stepsize=5e-4,
            sigma_regularization=1e0,
        )
    }
)
# preset_parameters.update({'clarity' : dict(sigma_regularization=1e1, deformative_stepsize=5e-1, linear_stepsize=2e-8, translational_stepsize=2e-5)}) # TODO: remove deprecated 'clarity' preset.


def get_registration_presets():
    """
    Get the names of all registration presets.

    Returns:
        dict_keys: The keys of the dictionary mapping all registration preset names to the corresponding registration parameters.
    """

    return preset_parameters.keys()


def get_registration_preset(preset: str) -> dict:
    """
    If <preset> is recognized, returns a dictionary containing the registration parameters corresponding to <preset>.

    Args:
        preset (str): The name of a preset keyed to a particular dictionary of registration parameters.

    Raises:
        NotImplementedError: Raised if <preset> is not a recognized preset name.

    Returns:
        dict: The registration kwargs specified by <preset>.
    """

    preset = preset.strip().lower()

    if preset in preset_parameters:
        return dict(
            preset_parameters[preset]
        )  # Recast as dict to provide a freely mutable copy.
    else:
        raise NotImplementedError(
            f"There is no preset for '{preset}'.\n"
            f"Recognized presets include:\n{list(preset_parameters.keys())}."
        )
