from utils.common import *
from functions.basic_functions import *

def generate_symmetric_tensor(k, sup_norm_bound=0.01, seed=42):
    """
    Generate a random symmetric k x k x k tensor with sup norm < sup_norm_bound.
    
    Parameters:
        k (int): Dimension of the tensor.
        sup_norm_bound (float): Upper bound for the supremum norm.
        seed (int): Random seed for reproducibility.
    
    Returns:
        jnp.ndarray: A symmetric k x k x k tensor.
    """
    key = random.PRNGKey(seed)

    # Generate a random tensor
    random_tensor = random.normal(key, (k, k, k))

    # Symmetrize the tensor
    sym_tensor = (random_tensor 
                  + jnp.transpose(random_tensor, (0, 2, 1)) 
                  + jnp.transpose(random_tensor, (1, 2, 0)) 
                  + jnp.transpose(random_tensor, (1, 0, 2)) 
                  + jnp.transpose(random_tensor, (2, 0, 1)) 
                  + jnp.transpose(random_tensor, (2, 1, 0))) / 6

    # Compute the supremum norm ‖E‖ := sup‖θ‖=1 |E(θ, θ, θ)|
    
    current_sup_norm = sup_norm(sym_tensor)

    # Scale the tensor to ensure sup norm is within bounds
    scaling_factor = sup_norm_bound / current_sup_norm if current_sup_norm > 0 else 0
    scaled_tensor = sym_tensor * scaling_factor

    return scaled_tensor

def generate_orthogonal_tensor(k, inf_norm=10, sup_norm=40, seed=42):
    """
    Generate a random orthogonally decomposable k*k*k tensor with given supremum norm.
    
    Parameters:
        k (int): Size of the tensor.
        sup_norm (float): Desired supremum norm of the tensor.
        seed (int): Random seed for reproducibility.
        
    Returns:
        tensor (jnp.ndarray): The generated orthogonally decomposable tensor.
        eigenvalues (jnp.ndarray): The eigenvalues used in the tensor decomposition.
        eigenvectors (jnp.ndarray): The orthonormal vectors used in the decomposition.
    """
    key = random.PRNGKey(seed)
    
    # Generate random orthonormal vectors
    random_matrix = random.normal(key, (k, k))
    eigenvectors = orth(random_matrix)  # Orthonormal vectors, shape (k, k)
    
    # Normalize eigenvectors to have unit norm (if necessary)
    eigenvectors = eigenvectors / jnp.linalg.norm(eigenvectors, axis=0, keepdims=True)
        
    # Generate random eigenvalues and scale to match the supremum norm
    random_key, subkey = random.split(key)
    eigenvalues = random.uniform(subkey, (k,), minval=inf_norm, maxval=sup_norm)
    
    # Construct the tensor
    tensor = jnp.zeros((k, k, k))
    for i in range(k):
        vi = eigenvectors[:, i]
        rank_one_tensor = jnp.einsum('i,j,k->ijk', vi, vi, vi)
        tensor += eigenvalues[i] * rank_one_tensor
    
    return tensor, eigenvalues, eigenvectors
