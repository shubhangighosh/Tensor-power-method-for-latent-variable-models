from utils.common import *
from functions.basic_functions import *

def tensor_power_iteration(tensor, L, N, seed=42):
    """
    Power iteration for symmetric tensors using JAX, parallelized over L initializations.

    Parameters:
        tensor (jnp.ndarray): Symmetric tensor of shape (k, k, k).
        L (int): Number of random initializations.
        N (int): Number of iterations for power iteration.

    Returns:
        best_theta (jnp.ndarray): Estimated eigenvector.
        lambda_val (float): Estimated eigenvalue.
        deflated_tensor (jnp.ndarray): Deflated tensor.
    """
    key = random.PRNGKey(seed)
    k = tensor.shape[0]
    best_theta = None
    best_value = -jnp.inf
    
    def power_iteration(theta, tensor, N):
        """Perform power iteration for a single initialization."""
        def cond_fn(state):
            theta, prev_theta, count = state
            norm_diff = jnp.linalg.norm(theta - prev_theta)
            return jax.lax.select(
                jax.lax.ge(norm_diff, 1e-6),  # Replacing >= with jax.lax.ge
                jax.lax.lt(count, N),          # Replacing < with jax.lax.lt
                False                          # equivalent to 'and' logic
            )

        def body_fn(state):
            theta, prev_theta, count = state
            updated_theta = jnp.tensordot(tensor, jnp.outer(theta, theta), axes=([1, 2], [0, 1]))
            theta = updated_theta / jnp.linalg.norm(updated_theta)
            return (theta, theta, count + 1)

        # Initialize with the current theta and prev_theta (both as the initial random guess)
        state = (theta, theta, 0)
        state = jax.lax.while_loop(cond_fn, body_fn, state)
        theta, _, _ = state
        return theta

    def compute_value(theta, tensor):
        """Compute the value associated with the tensor and theta."""
        return jnp.tensordot(tensor, jnp.einsum('i,j,k->ijk', theta, theta, theta), axes=([0, 1, 2], [0, 1, 2]))

    # Define the function to process a single initialization
    def single_iteration(theta):
        theta = theta / jnp.linalg.norm(theta)
        theta = power_iteration(theta, tensor, N)
        value = compute_value(theta, tensor)
        return value, theta
        
    def random_iteration(subkey):
        theta = random.normal(subkey, (k,))
        value, theta = single_iteration(theta)
        return value, theta

    # Parallelize the loop over L initializations
    keys = random.split(key, L)  # Generate L subkeys
    results = jax.vmap(random_iteration)(keys)  # Apply the single_iteration function to each subkey in parallel

    # Extract the results
    values, thetas = results
    best_value_idx = jnp.argmax(values)
    best_value = values[best_value_idx]
    best_theta = thetas[best_value_idx]

    # Perform final power iteration
    lambda_val, best_theta = single_iteration(best_theta)

    # Compute eigenvalue and deflation term
    # lambda_val = compute_value(best_theta, tensor)
    deflation_term = lambda_val * jnp.einsum('i,j,k->ijk', best_theta, best_theta, best_theta)
    deflated_tensor = tensor - deflation_term

    return best_theta, lambda_val, deflated_tensor

def top_k_tensor_eigenpairs(tensor, L, N, seed=42):
    """
    Compute the top k eigenvalues and eigenvectors of a symmetric tensor.

    Parameters:
        tensor (jnp.ndarray): Symmetric tensor of shape (n, n, n).
        k (int): Number of eigenpairs to compute.
        L (int): Number of random initializations for power iteration.
        N (int): Number of iterations for power iteration.

    Returns:
        eigenvalues (list): List of top k eigenvalues.
        eigenvectors (list): List of top k eigenvectors.
    """
    k = tensor.shape[0]
    eigenvalues = []
    eigenvectors = []
    deflated_tensor = tensor

    for i in range(k):
        top_theta, lambda_val, deflated_tensor = tensor_power_iteration(deflated_tensor, L, N, seed + i)
        eigenvalues.append(lambda_val)
        eigenvectors.append(top_theta)

    return eigenvalues, eigenvectors