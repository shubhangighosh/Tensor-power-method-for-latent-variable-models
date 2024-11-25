from utils.common import *
from functions.basic_functions import *
from functions.power_iterations import *
from bag_of_words.bow import *
from functions.error_functions import *

def main():
    L=1000000
    M=100000
    # Number of topics
    d = 7  # Number of distinct words in the vocabulary
    l = 4000  # Number of words in the document
    N = 10000  # Number of documents to generate
    k_values = range(3, 11)
    bow_errors_vs_k, bow_emp_errors_vs_k = compute_bow_errors_vs_k(k_values, L, M)

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, bow_errors_vs_k, color='b', label=r'Estimation error: $\frac{\|\mu_3 -\hat{M}_3\|}{\|\mu_3\|} $')
    plt.plot(k_values, bow_emp_errors_vs_k, color='r', label=r'Estimation error: $\frac{\|M_{3,o} -\hat{M}_{3,o}\|}{\|M_{3,o}\|} $')
    plt.xlabel('k (Tensor Dimension)', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.title('Bag-of-words Error Behavior with respect to Tensor Dimension k', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.savefig('bow_error_vs_k.png')
    plt.show()

if __name__ == "__main__":
    main()