from utils.common import *
from functions.basic_functions import *
from functions.power_iterations import *
from random_tensor_gen.generate import *
from functions.error_functions import *

def main():
    L=1000000
    M=100000
    # Define the range of k values
    k_values = range(3, 11)  # Example: Testing k from 3 to 20
    sup_norm_bound = 0.5
    # Compute errors
    errors_vs_k = compute_errors_vs_k(k_values, sup_norm_bound, L, M)
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, errors_vs_k, color='b')
    plt.xlabel('k (Tensor Dimension)', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.title('Error Behavior with respect to Tensor Dimension k', fontsize=16)
    plt.grid(True)
    plt.savefig('error_vs_k.png')
    plt.show()


if __name__ == "__main__":
    main()
