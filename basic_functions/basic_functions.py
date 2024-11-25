from utils.common import *

# Compute the supremum norm ‖E‖ := sup‖θ‖=1 |E(θ, θ, θ)|
def sup_norm(tensor, num_initializations=1000000, seed=42):
    """
    Maximize the supremum norm over multiple random initializations.

    Parameters:
        tensor (jnp.ndarray): Input tensor of shape (k, k, k).
        num_initializations (int): Number of random initializations to try.
        seed (int): Random seed for reproducibility.

    Returns:
        float: Maximum approximated supremum norm.
    """
    # Function to compute the supremum norm for a single random initialization
    def inter_sup_norm(tensor, key, k):
        theta = random.normal(key, (k,))
        theta /= jnp.linalg.norm(theta)  # Normalize to lie on the unit sphere
        return jnp.abs(jnp.tensordot(tensor, jnp.einsum('i,j,k->ijk', theta, theta, theta), axes=([0, 1, 2], [0, 1, 2])))

    # Generate random keys for the number of initializations
    key = random.PRNGKey(seed)
    keys = random.split(key, num_initializations)  # Generate multiple subkeys
    k = tensor.shape[0]

    # Use vmap to vectorize the sup_norm function over the keys
    sup_norms = jax.vmap(lambda subkey: inter_sup_norm(tensor, subkey, k))(keys)
    
    # Return the maximum value from the supremum norms
    return jnp.max(sup_norms)

def compute_error(tensor, eigenvalues, eigenvectors, seed=42):
    """
    Compute the supremum norm ‖E‖ := sup_{‖θ‖=1} |E(θ, θ, θ)|.

    Parameters:
        tensor (jnp.ndarray): Original tensor of shape (n, n, n).
        eigenvalues (list): List of eigenvalues (λ_j) of length k.
        eigenvectors (list): List of eigenvectors (v_j) of shape (k, n).
        max_iter (int): Maximum number of iterations for power iteration.
        tol (float): Tolerance for convergence.
        seed (int): Random seed for initialization.

    Returns:
        float: Supremum norm of the error tensor.
    """
    n = tensor.shape[0]
    k = len(eigenvalues)

    # Reconstruct the tensor from eigenpairs
    reconstructed_tensor = jnp.zeros_like(tensor)
    for j in range(k):
        v_j = eigenvectors[j]
        λ_j = eigenvalues[j]
        v_j_outer = jnp.einsum('i,j,k->ijk', v_j, v_j, v_j)  # Compute v_j^⊗3
        reconstructed_tensor += λ_j * v_j_outer

    # Compute the error tensor
    error_tensor = tensor - reconstructed_tensor   

    # Compute the supremum norm
    op_norm = sup_norm(error_tensor)

    return op_norm