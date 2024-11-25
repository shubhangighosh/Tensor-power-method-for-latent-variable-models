from utils.common import *
from functions.basic_functions import *
from functions.power_iterations import *
from random_tensor_gen.generate import *
from bag_of_words.bow import *

def tens_vs_errmat(sup_norm_bound, L, M):
    """
    Compute the correspondence between estimated eigenvectors and original eigenvectors,
    and return the error along with the inner products of corresponding eigenvectors.

    Parameters:
        sup_norm_bound (float): Supremum norm bound for noise tensor.
        L (int): Number of random initializations for power iteration.
        M (int): Number of iterations for power iteration.

    Returns:
        err (float): Supremum norm of the error tensor.
        correspondences (list): List of tuples indicating correspondence (original_idx, estimated_idx).
        inner_products (list): List of inner products for corresponding eigenvectors.
    """
    # Generate the symmetric tensor with the current sup_norm_bound
    etens = generate_symmetric_tensor(k=3, sup_norm_bound=sup_norm_bound)

    # Generate the orthogonal tensor and the combined tensor
    otens, oeval, oevecs = generate_orthogonal_tensor(k=3)

    # Combine the tensors
    exptens = otens + etens

    # Compute the top k eigenpairs for the combined tensor
    eigenvalues, eigenvectors = top_k_tensor_eigenpairs(exptens, L, M)

    # Normalize all eigenvectors (estimated and original) to unit norm
    oevecs = oevecs / jnp.linalg.norm(oevecs, axis=0, keepdims=True)
    eigenvectors = jnp.array(eigenvectors)  # Convert to JAX array
    eigenvectors = eigenvectors / jnp.linalg.norm(eigenvectors, axis=0, keepdims=True)

    # Compute the correspondence based on inner products
    k = oevecs.shape[1]
    inner_product_matrix = jnp.abs(jnp.dot(oevecs.T, eigenvectors))  # Shape (k, k), absolute value of inner products
    correspondences = []
    inner_products = []

    for i in range(k):
        # Find the estimated eigenvector with the largest inner product for the i-th original eigenvector
        max_idx = jnp.argmax(inner_product_matrix[i])
        max_inner_product = inner_product_matrix[i, max_idx]

        # Store correspondence and the inner product
        correspondences.append((i, max_idx))
        inner_products.append(max_inner_product)

        # Set the row and column to 0 to avoid duplicate assignment
        inner_product_matrix = inner_product_matrix.at[i, :].set(0)
        inner_product_matrix = inner_product_matrix.at[:, max_idx].set(0)

    # Compute the error (supremum norm of the error tensor)
    err = compute_error(otens, eigenvalues, eigenvectors)

    return err, correspondences, inner_products

def compute_errors_vs_k(k_values, sup_norm_bound, L, M, seed=42):
    """
    Compute the errors for different tensor dimensions (k x k x k).
    
    Parameters:
        k_values (list or array): Range of k values to test.
        sup_norm_bound (float): Supremum norm bound for the noise tensor.
        L (int): Number of random initializations for power iteration.
        M (int): Number of iterations for power iteration.
        seed (int): Random seed for reproducibility.

    Returns:
        errors_vs_k (list): Errors corresponding to each k value.
    """
    errors_vs_k = []

    for k in tqdm(k_values, desc="Computing Errors for different k"):
        # Generate the symmetric tensor with the current k and sup_norm_bound
        etens = generate_symmetric_tensor(k=k, sup_norm_bound=sup_norm_bound, seed=seed)

        # Generate the orthogonal tensor and the combined tensor
        otens, oeval, oevecs = generate_orthogonal_tensor(k=k)

        # Combine the tensors
        exptens = otens + etens

        # Compute the top k eigenpairs for the combined tensor
        eigenvalues, eigenvectors = top_k_tensor_eigenpairs(exptens, L, M, seed=seed)

        # Compute the error (supremum norm of the error tensor)
        error = compute_error(otens, eigenvalues, eigenvectors, seed=seed)

        errors_vs_k.append(error)

    return errors_vs_k

def compute_bow_errors_vs_k(k_values, L, M, seed=42):
    """
    Compute the errors for different tensor dimensions (k x k x k).
    
    Parameters:
        k_values (list or array): Range of k values to test.
        sup_norm_bound (float): Supremum norm bound for the noise tensor.
        L (int): Number of random initializations for power iteration.
        M (int): Number of iterations for power iteration.
        seed (int): Random seed for reproducibility.

    Returns:
        errors_vs_k (list): Errors corresponding to each k value.
    """
    errors_vs_k = []
    emp_errors_vs_k = []

    for k in tqdm(k_values, desc="Computing Errors for different k"):
        # Regenerate the topic distribution (w) and word distributions (mu)
        alpha_w = jnp.ones(k)
        w = jax.random.dirichlet(jax.random.PRNGKey(0), alpha_w)
        mu = jax.random.dirichlet(jax.random.PRNGKey(1), jnp.ones(d), shape=(k,))
    
        # Regenerate documents
        base_key = jax.random.PRNGKey(seed)  # Base key for randomness
        keys = jax.random.split(base_key, N)
        generate_fn = jax.vmap(lambda key: generate_document(w, mu, l, key))
        documents, topics, _ = generate_fn(keys)
        documents = jnp.array(documents, dtype=jnp.float64)
    
        # Compute M2 and M3
        M2, M3 = compute_moments(documents)
    
        # Compute M3hat
        W, M3hat = compute_M3hat(M2, M3, k)

        # Compute the top k eigenpairs for the combined tensor
        eigenvalues, eigenvectors = top_k_tensor_eigenpairs(M3hat, L, M, seed=seed)
        eigenvalues = jnp.array(eigenvalues)
        eigenvectors = jnp.array(eigenvectors)
        true_M3 = jnp.einsum('l,li,lj,lk->ijk', w, mu, mu, mu)
        est_M3_hat = jnp.einsum('l,li,lj,lk->ijk', eigenvalues, eigenvectors, eigenvectors, eigenvectors)
        est_M3 = jnp.einsum('ijk,pi,qj,rk->pqr', est_M3_hat, W, W, W)
        # Compute the error tensor
        error_tensor = true_M3 - est_M3  
        emp_error_tensor = M3hat - est_M3_hat
    
        # Compute the supremum norm
        op_norm = sup_norm(error_tensor)/sup_norm(true_M3)
        est_err_norm = sup_norm(emp_error_tensor)/sup_norm(M3hat)
        errors_vs_k.append(op_norm)
        emp_errors_vs_k.append(est_err_norm)

    return errors_vs_k, emp_errors_vs_k

