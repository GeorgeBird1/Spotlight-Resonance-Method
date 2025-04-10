import numpy as np
import itertools


def spotlight_resonance_method(*,
                               latent_layer_activations: np.array,
                               privileged_basis: np.array,
                               epsilon: float = 0.9,
                               max_planes: int = None,
                               perm_or_comb: str = "PERM",
                               angular_resolution: int = 100,
                               verbose: bool = False,
                               ):
    """
    Performs the spotlight resonance method on the activations provided.

    :param latent_layer_activations: These are a set of activations from the latent layer of interest. Numpy array of
    shape [number of samples, dimensionality of layer]

    :param privileged_basis: These are the set of privileged basis directions for the layer. Numpy array of shape
    [number of privileged basis vectors, dimensionality of layer]

    :param epsilon: This is the epsilon parameter for spotlight resonance, it effectively is a tolerance which represents a
    cone angle (phi=arccos epsilon) about the 'probe vector' which counts the amount of activations inside. (-1<epsilon<1)

    :param max_planes: There is a quadratic growth in privileged bivectors with each new privileged vector. This sets an upper
    limit on the number of bivectors considered (drawn randomly). If set to None, then it uses all bivectors.

    :param perm_or_comb: "PERM" or "COMB" represents the permuation or combination form of SRM technique.

    :param angular_resolution: Determines how many angles SRM will be evaluated for. For example 720,
    would result in calculations for each 0.5 degrees about a full rotation.

    :param verbose: Boolean which is set to true outputs progress of SRM.

    :return: Two numpy arrays.
        angles: The angles used in the calculation (shape [angular resolution])
        SRM values: The result of the SRM calculation on the activations provided
                    the shape of this array is [number of bivectors, angular resolution]
    """

    # Basic checks
    if perm_or_comb.upper() not in ["PERM", "COMB"]: raise ValueError(f"Please ensure \"PERM\" or \"COMB\" SRM is selected, not {perm_or_comb=}.")
    if epsilon>1 or epsilon<-1: raise ValueError(f"Please ensure \"epsilon\" is in range [-1, 1] not {epsilon=}.")
    if privileged_basis.shape[1] != latent_layer_activations.shape[1]: raise ValueError("Privileged basis invalid, must be same dimensionality as the activations provided!")

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
            print(f"Total planes={len(valid_plane_indices)} but only a random sample of {max_planes} will be calculated.")
        else:
            print(f"All {len(valid_plane_indices)} will be calculated.")
        valid_plane_indices = valid_plane_indices[:max_planes]

    # The angles to iterate through
    angles = np.linspace(0, np.pi * 2, angular_resolution)

    srm_values = np.zeros((len(valid_plane_indices), angular_resolution))
    # Iterate over the various planes to compute SRM values
    for p, indices in enumerate(valid_plane_indices):
        if verbose: print(f"Calculating plane {p+1} of {len(valid_plane_indices)}.")
        # Lie generator for rotation in this plane
        rotation_generator = vectors_to_bivectors(
            privileged_basis[indices[0], :],
            privileged_basis[indices[1], :]
        )
        # Then produce all the rotation matrices required.
        rotation_matrices = generate_special_orthogonal_matrices(rotation_generator, angles)

        # The probe vector is one of the vectors which lie within the current bivector. For simplicity the first
        # vector used to construct the bivector is chosen.
        probe_vector = privileged_basis[indices[0], :]

        # Then perform spotlight resonance method. Can replace this method with signed spotlight resonance if desired.
        srm_values[p, :] = f_spotlight_resonance(latent_layer_activations, probe_vector, rotation_matrices, epsilon)
    return angles, srm_values


def vectors_to_bivectors(vector1, vector2):
    """
    Takes two vectors and constructs a representation of their bivector
    :param vector1: Numpy array of shape [dimensionality of layer]
    :param vector2: Numpy array of shape [dimensionality of layer]
    :return: representation of bivector. np.array of shape [dimensionality of layer, dimensionality of layer]
    """
    # Calculate the exterior product of normalised vectors (should be norm 1 anyway).
    outer_product = np.einsum("i, j->ij", normalise(vector1, axis=0), normalise(vector2, axis=0))
    # Calculate the bivector representation (rotation generator)
    rotation_generator = 0.5*(outer_product - outer_product.T)
    return rotation_generator

def hermitian_conjugate(array):
    """
    Perform the hermitian conjugate on a numpy array
    :param array: Complexified numpy array of shape [a, b]
    :return: Hermitian conjugate of numpy array of shape [b, a]
    """
    return np.conj(array).T

def generate_special_orthogonal_matrices(generator, angles):
    """
    Produces a set of special orthogonal matrices from the specified generator and angles.

    :param generator: This is the bivector produce from the privileged vectors. It is of shape:
    [dimensionality of layer, dimensionality of layer]

    :param angles: The angles used to generate special orthogonal matrices [angular_resolution].

    :return: Special orthogonal matrices of shape [angular_resolution, dimensionality of layer, dimensionality of layer]
    """
    # To perform exponentiation of the bivector, the eigendecomposed identity discussed in paper is useful.
    vals, vecs = np.linalg.eig(generator)

    # This steps normalises the bivector's eigenvalue to i as described in SRM paper
    vals = 1j * vals.imag
    normalisation = np.abs(vals).max()
    vals /= normalisation

    # Perform calculation discussed in paper, first multiply eigenvalues by the angles and exponentiate
    exp_eigenvalues = np.exp(np.einsum("a, v->av", angles, vals))
    # Sandwich those exponentiated eigenvalues between the eigenvectors and hermitian conjugate of eigenvectors (identity)
    rotation_matrix = np.einsum("ij, aj, jk->aik", vecs, exp_eigenvalues, hermitian_conjugate(vecs))
    return rotation_matrix.real

def normalise(array, axis):
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return array / norms

def f_spotlight_resonance(activations, probe_vector, rotation_matrices, epsilon):
    """
    Performs the spotlight resonance function on the provided activations.

    :param activations: These are a set of activations from the latent layer of interest. Numpy array of
    shape [number of samples, dimensionality of layer]

    :param probe_vector: This is the vector which the spotlight cone is centered about. Numpy array of shape [dimensionality of layer]

    :param rotation_matrices: Special orthogonal matrices of shape [angular_resolution, dimensionality of layer, dimensionality of layer]

    :param epsilon: This is the epsilon parameter for spotlight resonance, it effectively is a tolerance which represents a
    cone angle (phi=arccos epsilon) about the 'probe vector' which counts the amount of activations inside. (-1<epsilon<1)

    :return: This is the result of the SRM calculation and is a numpy array of shape [angular_resolution]
    """
    # Normalised inner product calculation
    normalised_inner_product = np.einsum("bi, j, aij->ba", normalise(activations, axis=1), probe_vector, rotation_matrices)

    # Points within cone
    new_similarities = np.zeros_like(normalised_inner_product)
    new_similarities[normalised_inner_product >= epsilon] += 1
    return new_similarities.mean(axis=0)

def f_signed_spotlight_resonance(activations, probe_vector, rotation_matrices, epsilon):
    """
    Performs the SIGNED spotlight resonance function on the provided activations.

    :param activations: These are a set of activations from the latent layer of interest. Numpy array of
    shape [number of samples, dimensionality of layer]

    :param probe_vector: This is the vector which the spotlight cone is centered about. Numpy array of shape [dimensionality of layer]

    :param rotation_matrices: Special orthogonal matrices of shape [angular_resolution, dimensionality of layer, dimensionality of layer]

    :param epsilon: This is the epsilon parameter for spotlight resonance, it effectively is a tolerance which represents a
    cone angle (phi=arccos epsilon) about the 'probe vector' which counts the amount of activations inside. (-1<epsilon<1)

    :return: This is the result of the signed-SRM calculation and is a numpy array of shape [angular_resolution]
    """
    # Normalised inner product calculation
    normalised_inner_product = np.einsum("bi, j, aij->ba", normalise(activations, axis=1), probe_vector, rotation_matrices)

    # Points within cone
    new_similarities = np.zeros_like(normalised_inner_product)
    new_similarities[normalised_inner_product >= epsilon] += 1
    new_similarities[normalised_inner_product <= -epsilon] -= 1
    return new_similarities.mean(axis=0)
