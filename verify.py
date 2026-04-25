import numpy as np
import os


def read_matrix(filename):
    with open(filename, 'r') as f:
        n = int(f.readline())
        return np.array([list(map(float, f.readline().split())) for _ in range(n)])


sizes = [200, 400, 800, 1200, 1600, 2000]
threads = [1, 2, 4, 8, 12]

print("=" * 50)
print("OPENMP VERIFICATION")
print("=" * 50)

passed = 0
total = 0

for n in sizes:
    folder = f"data_{n}"

    if not os.path.exists(folder):
        print(f"{folder} not found")
        continue

    A = read_matrix(f"{folder}/A.txt")
    B = read_matrix(f"{folder}/B.txt")
    C_ref = np.dot(A, B)

    for t in threads:
        filename = f"{folder}/C_{t}.txt"
        if not os.path.exists(filename):
            continue

        C_omp = read_matrix(filename)
        diff = np.max(np.abs(C_omp - C_ref))
        total += 1

        if diff < 1e-6:
            print(f"{n}x{n}, {t} PASSED (diff={diff:.2e})")
            passed += 1
        else:
            print(f"{n}x{n}, {t} NOT PASSED (diff={diff:.2e})")

print("=" * 50)
print(f"RESULTS: {passed}/{total}")
if passed == total:
    print("ALL CHECKS PASSED")
print("=" * 50)