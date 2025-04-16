

import random, sys, math
import numpy as np

def generate_uniform_matrix(n: int, a: float, b: float) -> np.ndarray:
    '''
    Generates an N x N matrix of random values sampled from a uniform ditribution
    of the range [a,b]

    Args:
        n (int): The length of rows and columns for the matrix
        a (float): The beginning of the range
        b (float): The end of the range

    Returns
        (numpy.ndarray): Generated matrix
    '''
    if (a >= b): raise ValueError("b must be greater than a")

    matrix = []
    for i in range(0, n):
        matrix.append([])
        for j in range(0, n):
            uniform_val = random.uniform(a, b)
            matrix[i].append(uniform_val)
    return np.array(matrix)



def print_matrix(m: np.ndarray, f = sys.stdout):
    '''
    Prints out a given array into a more readable format to a given file output

    Args:
        m (numpy.ndarray): Matrix to print
        f (file): File object to write to. Normally sys.stdout
    '''
    print("PRINTING!!")
    f.write("[\n")
    for row in m:
        row_str = "\t["
        for x in row:
            row_str += str(x) + ", "
        row_str = row_str[:-2]
        f.write(row_str + "],\n")
    f.write("]\n")


def naive_matrix_multiplication(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    
    if cols_A != rows_B:
        raise ValueError("Columns of A must be equal to rows of B")
    
    # Initialize result matrix with zeros
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    # Naive triple loop for matrix multiplication
    for i in range(rows_A):
        print(f'Row {i} or {rows_A}')
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

def FPO_operations_naive(n):
    multiplications = n * n * n  # Each element requires n multiplications
    additions = n * n * (n - 1)  # Each element requires (n-1) additions
    return multiplications + additions

def FPO_operations_strassen(n):
    if n == 1:
        return 1  # Base case: single multiplication
    
    multiplications = 7 * FPO_operations_strassen(n // 2)  # Strassen uses 7 recursive calls
    additions = 18 * (n // 2) ** 2  # Strassen requires 18 matrix additions/subtractions
    
    return multiplications + additions

def strassen_matrix_multiplication(A, B):
    if (A.shape != B.shape):
        raise ValueError("Matrices must be square and of the same size")

    n = A.shape[0]
    # print(n);

    if n == 1:
        return (A @ B)
    
    mid = n // 2
    A11, A12, A21, A22 = A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]
    B11, B12, B21, B22 = B[:mid, :mid], B[:mid, mid:], B[mid:, :mid], B[mid:, mid:]
    
    M1 = strassen_matrix_multiplication(A11 + A22, B11 + B22)
    M2 = strassen_matrix_multiplication(A21 + A22, B11)
    M3 = strassen_matrix_multiplication(A11, B12 - B22)
    M4 = strassen_matrix_multiplication(A22, B21 - B11)
    M5 = strassen_matrix_multiplication(A11 + A12, B22)
    M6 = strassen_matrix_multiplication(A21 - A11, B11 + B12)
    M7 = strassen_matrix_multiplication(A12 - A22, B21 + B22)
    
    C11 = np.array(M1) + np.array(M4) - np.array(M5) + np.array(M7)
    C12 = np.array(M3) + np.array(M5)
    C21 = np.array(M2) + np.array(M4)
    C22 = np.array(M1) - np.array(M2) + np.array(M3) + np.array(M6)
    
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    return C.tolist()
  

# A = np.array([
#     [1,2],
#     [3,4]
#     ])

# result = naive_matrix_multiplication(A,A)
# print_matrix(result)

# result = strassen_matrix_multiplication(A,A)
# print_matrix(result)

f = open("exam2-results.txt", "w+")

A = generate_uniform_matrix(int(math.pow(2,10)), -2, 2)
print(A)
print_matrix(A, f)
B = generate_uniform_matrix(int(math.pow(2,10)), -2, 2)
print(B)
print_matrix(B, f)


# print(f"Number of FP operations for naive: {FPO_operations_naive(math.pow(2, 10))}")
# print(f"Number of FP operations for strassen: {FPO_operations_strassen(math.pow(2, 10))}")

# naive_result = naive_matrix_multiplication(A,B)
# print_matrix(naive_result)
s_result = strassen_matrix_multiplication(A,B)
print_matrix(s_result, f)

f.close()
# result = strassen_matrix_multiplication(A,A)
# print_matrix(result)