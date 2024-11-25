from utils.common import *
from functions.basic_functions import *
from functions.power_iterations import *
from random_tensor_gen.generate import *
from functions.error_functions import *

def main():
    L=1000000
    M=100000

    # Define the range of sup_norm_bound values
    sup_norm_bounds = np.linspace(0.001, 45, 500)  # You can adjust the range and the number of points
    
    # Now, vectorize the computation of errors using vmap
    # Initialize a list to store the errors
    errors = []
    all_inner_products = []
    
    # Loop over each element in sup_norm_bounds
    for x in sup_norm_bounds:
        error, _, inner_products = tens_vs_errmat(x, L, M)  # Compute error for the current element
        errors.append(error)  # Append the result to the errors list
        all_inner_products.append(inner_products)
    
    
    # Convert the list of errors to a JAX array if needed
    errors = jnp.array(errors)
    all_inner_products = jnp.array(all_inner_products)
    # Sort each row of all_inner_products in descending order
    sorted_inner_products = jnp.sort(all_inner_products, axis=1)[:, ::-1]

    plt.figure(figsize=(8, 6))
    plt.plot(sup_norm_bounds, errors, label='Error vs. sup_norm_bound', color='b')
    plt.xlabel(r'Sup Norm Bound $\|E\|$', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.title('Behavior of Error with respect to Sup Norm Bound', fontsize=16)
    plt.grid(True)
    plt.savefig('error_vs_sup_norm_bound.png')
    plt.show()

    # Plot the inner products
    plt.figure(figsize=(8, 6))
    for i in range(sorted_inner_products.shape[1]):  # Loop through each eigenvector index
        plt.plot(sup_norm_bounds, sorted_inner_products[:, i], label=f'Inner Product {i+1}')
    plt.xlabel('Sup Norm Bound', fontsize=14)
    plt.ylabel('Inner Product', fontsize=14)
    plt.title('Behavior of Inner Products with respect to Sup Norm Bound', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.savefig('ipdt_vs_sup_norm_bound.png')
    plt.show()

if __name__ == "__main__":
    main()

