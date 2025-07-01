import numpy as np
import itertools


def PrivilegedPlaneProjectiveMethod(*,
                                    latent_layer_activations: np.array,
                                    privileged_basis: np.array,
                                    epsilon: float = 0.9,
                                    max_planes: int = None,
                                    perm_or_comb: str = "PERM",
                                    verbose: bool = False,
                                    ):
    """
    Performs the Privileged Plane Projective Method method on the activations provided. Activations within the specified angle are projected onto the plane. There in-plane coordinates are then outputted for the user to analyse.

    :param latent_layer_activations: These are a set of activations from the latent layer of interest. Numpy array of
    shape [number of samples, dimensionality of layer]

    :param privileged_basis: These are the set of privileged basis directions for the layer. Numpy array of shape
    [number of privileged basis vectors, dimensionality of layer]

    :param epsilon: Uses the epsilon parameter similar to the spotlight resonance method, it effectively is a tolerance which represents a cone angle (phi=arccos epsilon) about the 'probe vector' which counts the amount of activations inside. (-1<epsilon<1)).  (or alternative method commented out  Gives a tolerance for the norm of the vector component perpendicular to the plane. In effect, if the original vector is sufficiently close to the plane (small perpendicular component), then it is projected to the plane, if not,discarded.)

    :param max_planes: There is a quadratic growth in privileged bivectors with each new privileged vector. This sets an upper limit on the number of bivectors considered (drawn randomly). If set to None, then it uses all bivectors.

    :param perm_or_comb: "PERM" or "COMB" represents the permuation or combination form of SRM technique.

    :param verbose: Boolean which is set to true outputs progress of SRM.

    :return: One numpy array containing in-plane coordinates for all activations which are suitably aligned within the epsilon tolerance to the privileged plane [number of points, in plane coordinates]
    """

    # Basic checks
    if perm_or_comb.upper() not in ["PERM", "COMB"]: raise ValueError(
        f"Please ensure \"PERM\" or \"COMB\" SRM is selected, not {perm_or_comb=}.")
    if epsilon > 1 or epsilon < -1: raise ValueError(f"Please ensure \"epsilon\" is in range [-1, 1] not {epsilon=}.")
    if privileged_basis.shape[1] != latent_layer_activations.shape[1]: raise ValueError(
        "Privileged basis invalid, must be same dimensionality as the activations provided!")

    # Indexes privileged vectors
    basis_indices = list(range(privileged_basis.shape[0]))

    # Then produces all valid privileged bivectors
    if verbose: print("Calculating Bivectors... ", end="")
    if perm_or_comb.upper() == "PERM":
        valid_plane_indices = list(itertools.permutations(basis_indices, 2))
    else:
        valid_plane_indices = list(itertools.combinations(basis_indices, 2))

    # (Optional) shuffle these, such that when a maximum amount are chosen they are chosen randomly.
    np.random.shuffle(valid_plane_indices)
    if verbose: print("Done!")

    if max_planes is not None:
        if max_planes < len(valid_plane_indices):
            print(
                f"Total planes={len(valid_plane_indices)} but only a random sample of {max_planes} will be calculated.")
        else:
            print(f"All {len(valid_plane_indices)} will be calculated.")
        valid_plane_indices = valid_plane_indices[:max_planes]

    # Create an array to store projections
    projected_points = np.zeros((0, 2))

    # Iterate over the various planes to compute projected bivector values
    for p, indices in enumerate(valid_plane_indices):
        if verbose: print(f"Calculating plane {p + 1} of {len(valid_plane_indices)}.")

        # Find the two (unit) basis vectors defining the plane
        ehat1 = normalise(privileged_basis[indices[0], :], axis=0)
        ehat2 = normalise(privileged_basis[indices[1], :], axis=0)

        # Requires the pseudo inverse of the plane's basis vectors to determine the scaling of each basis component
        try:
            # Find ehat2_prime which is perpendicular to ehat1 and has a positive dot product with ehat2
            dot = np.einsum("i, i->", ehat1, ehat2)
            ehat2_prime = normalise(ehat2 - dot * ehat1, axis=0)
            E = np.array([ehat1, ehat2_prime])
            pseudo_inverse = E.T @ np.linalg.inv(E @ E.T)
        except:
            if verbose: print("Parallel - Skipped!")
            continue
        # Determine components
        components = latent_layer_activations @ pseudo_inverse

        # Calculate the vector within the plane of interest.
        in_plane_vector = np.einsum("se, ed->sd", components, E)

        # Calculate the angle between the plane and the original vector
        angle_to_plane = np.linalg.norm(in_plane_vector, axis=1) / np.linalg.norm(latent_layer_activations, axis=1)

        # If that angle is within phi then add then add the inplane vector to the stack of projected points
        within_angle = angle_to_plane >= epsilon

        # Alternative method, using a cutoff on the length of the perpendicular component instead of angle.
        # plane_perpendicular_vector_norm = np.linalg.norm(latent_layer_activations-in_plane_vector, axis=1)
        # within_tolerance = plane_perpendicular_vector_norm < epsilon

        # Stack the components within the plane
        projected_points = np.vstack([projected_points, components[within_angle, :]])

    return projected_points


def normalise(array, axis):
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return array / norms
