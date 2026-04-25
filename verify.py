import numpy as np
import os


def read_matrix(filename):
    """Read matrix from file"""
    try:
        with open(filename, 'r') as f:
            n = int(f.readline().strip())
            matrix = []
            for _ in range(n):
                row = list(map(float, f.readline().split()))
                matrix.append(row)
        return np.array(matrix)
    except Exception as e:
        print(f"reading error {filename}: {e}")
        return None


sizes = [200, 400, 800, 1200, 1600, 2000]


print("Results")


all_passed = True
passed_count = 0

for n in sizes:
    folder = f"data_{n}"

    if not os.path.exists(folder):
        print(f"folder {folder} not found")
        continue

    A = read_matrix(f"{folder}/A.txt")
    B = read_matrix(f"{folder}/B.txt")
    C_cpp = read_matrix(f"{folder}/C.txt")

    if A is None or B is None or C_cpp is None:
        print(f"Size {n}x{n}: cannot read files")
        all_passed = False
        continue

    C_ref = np.dot(A, B)

    diff = np.max(np.abs(C_cpp - C_ref))

    if diff < 1e-6:
        print(f"PASSED Size {n}x{n}: max difference = {diff:.2e}")
        passed_count += 1
    else:
        print(f"NOT PASSED Size {n}x{n}: max difference = {diff:.2e}")
        all_passed = False

        print(f"   First element C_cpp[0,0] = {C_cpp[0, 0]:.6f}")
        print(f"   First element C_ref[0,0] = {C_ref[0, 0]:.6f}")
        print(f"   First element A[0,0] = {A[0, 0]:.6f}")
        print(f"   First element B[0,0] = {B[0, 0]:.6f}")


print(f"Verification: {100.*passed_count/len(sizes)}% passed")
if all_passed:
    print("ALL RESULTS PASSED VERIFICATION")
else:
    print("NOT ALL RESULTS PASSED VERIFICATION")
