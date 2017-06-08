from numba import vectorize, cuda
import numpy as np
from timeit import default_timer as timer


def vector_add(a, b, c):
    for i in range(0, len(a)):
        c[i] = a[i] + b[i]


@cuda.jit
def add_cuda(a, b, c):
    i = cuda.grid(1)
    if i < len(a):
        c[i] = a[i] + b[i]


def main():
    N = 1000000

    a = np.ones(N, dtype=np.int)
    b = np.ones(N, dtype=np.int)
    c = np.zeros(N, dtype=np.int)

    classic_start = timer()
    vector_add(a, b, c)
    classic_end = timer()

    print("Time with classic function: %f seconds" % (classic_end - classic_start))

    threads_per_block = 1024
    block_per_grid = (N / threads_per_block) + (N % threads_per_block)

    cuda_start = timer()
    d_a = cuda.to_device(np.ones(N, dtype=np.int))
    d_b = cuda.to_device(np.ones(N, dtype=np.int))
    d_c = cuda.to_device(np.zeros(N, dtype=np.int))
    add_cuda[block_per_grid, threads_per_block](d_a, d_b, d_c)
    local = d_c.copy_to_host()
    cuda_end = timer()




    # print(local[4])

    print("Time with CUDA function: %f seconds" % (cuda_end - cuda_start))

if __name__ == '__main__':
    main()






