from utils.common import *

# Function to simulate a document with random topic and word distributions
def generate_document(w, mu, l, key):
    # Step 1: Draw the topic h according to w (topic distribution)
    key, subkey = jax.random.split(key)
    h = jax.random.choice(subkey, a=w.shape[0], p=w)  # Sample topic
    
    # Step 2: For each word in the document, draw from the multinomial distribution based on topic h
    key, subkey = jax.random.split(key)
    words = jax.random.choice(subkey, a=d, shape=(l,), p=mu[h])  # Sample words
    
    # Step 3: Convert words into one-hot vectors (d-dimensional)
    one_hot_words = jnp.eye(d)[words]  # One-hot encoding for words
    
    return one_hot_words, h, key  # Return the one-hot encoded document, the topic, and updated key

def compute_moments(documents):
    """
    Computes the second (M2) and third (M3) moments in a parallelized fashion.
    
    Parameters:
        documents (list of jnp.ndarray): List of 2D arrays where each row is a one-hot vector
                                          representing a word in a document.
    
    Returns:
        M2 (jnp.ndarray): Second moment matrix of shape (d, d).
        M3 (jnp.ndarray): Third moment tensor of shape (d, d, d).
    """
    d = documents[0].shape[1]  # Dimensionality of one-hot vectors
    N = len(documents)         # Total number of documents
    l = documents[0].shape[0]  # Maximum number of words in a document (assumed constant here)

    # Function to compute per-document moments
    def per_document_moments(doc):
        doc /= l
        # Compute M2 (word pairs)
        pairwise_m2 = jnp.einsum('ij,lk->jk', doc, doc) - jnp.einsum('ij,ik->jk', doc, doc) # All pairwise outer products
        # Compute M3 (word triplets)
        triplets_m3 = jnp.einsum('il,jm,kn->lmn', doc, doc, doc) - jnp.einsum('il,im,kn->lmn', doc, doc, doc) - jnp.einsum('il,jm,jn->lmn', doc, doc, doc)- jnp.einsum('il,jm,in->lmn', doc, doc, doc) + 2*jnp.einsum('il,im,in->lmn', doc, doc, doc)# All triplet outer products
        return pairwise_m2, triplets_m3

    # Vectorized computation over all documents
    per_doc_m2, per_doc_m3 = jax.vmap(per_document_moments)(documents)

    # Aggregate moments over all documents
    M2 = jnp.sum(per_doc_m2, axis=0, dtype=jnp.float64)
    M3 = jnp.sum(per_doc_m3, axis=0, dtype=jnp.float64)

    # Normalize by the total number of word pairs and triplets
    # total_pairs = jnp.int64(N) * (jnp.int64(l) * (jnp.int64(l) - 1)) 
    # total_triplets = jnp.int64(N) * (jnp.int64(l) * (jnp.int64(l) - 1) * (jnp.int64(l) - 2))
    M2 /= jnp.float64(N)
    M3 /= jnp.float64(N)

    
    return M2, M3

def compute_M3hat(M2, M3, k):
    # Step 1: Eigen decomposition of M2 to find the transformation W
    M2 += 0.00001*jnp.eye(M2.shape[0])
    eigenvalues, eigenvectors = jnp.linalg.eigh(M2)  # M2 is assumed to be symmetric

    # Step 2: Use eigenvectors as the linear transformation W (d x k matrix)
    # We assume that the eigenvectors are sorted in increasing order of eigenvalue
    # Take the first k eigenvectors to form W
    W = eigenvectors[:, :k]  # d x k
    # Step 3: Normalize W such that W^T M2 W = I
    # Since the eigenvectors are already orthonormal, no additional scaling is needed

    # Step 4: Compute the transformed M3 (M3hat)
    # M3 is assumed to be a d x d x d tensor. We apply the transformation.
    M3hat = jnp.tensordot(M3, W, axes=([2], [0]))  # Transform first dimension
    M3hat = jnp.tensordot(M3hat, W, axes=([1], [0]))  # Transform second dimension
    M3hat = jnp.tensordot(M3hat, W, axes=([0], [0]))  # Transform third dimension
    return W, M3hat
