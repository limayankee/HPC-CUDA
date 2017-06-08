import cv2
import math
from numba import cuda
import numpy as np
from timeit import default_timer as timer

@cuda.jit
def conv_inv_cuda(img_i):
    i, j, k = cuda.grid(3)
    if i < img_i.shape[0] and j < img_i.shape[1]:
        img_i[i][j][k] = 255 - img_i[i][j][k]


def conv_inv(img_i):
    for i in range(0, img_i.shape[0]):
        for j in range(0, img_i.shape[1]):
            for k in range(0, img_i.shape[2]):
                img_i[i][j][k] = 255 - img_i[i][j][k]



def main():
    img = np.asarray(cv2.imread('img2.jpg'))

    print('Image size: %i X %i' % (img.shape[0], img.shape[1]))

    d_img = cuda.to_device(img)

    threadsPerBlock = (18, 18, 3)  # 1024 treads per block, that's the max for our case
    blockspergrid_x = int(math.ceil(img.shape[0] / threadsPerBlock[0])) + (img.shape[0] % threadsPerBlock[0])
    blockspergrid_y = int(math.ceil(img.shape[1] / threadsPerBlock[1])) + (img.shape[1] % threadsPerBlock[1])
    blockspergrid_z = 1
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    CUDA_start = timer()
    conv_inv_cuda[blockspergrid, threadsPerBlock](d_img)
    CUDA_end = timer()

    classic_start = timer()
    conv_inv(img)
    classic_end = timer()

    print('Temps pour invertion des couleurs d\'une image')
    print('CUDA: %f secondes' % (CUDA_end - CUDA_start))
    print('classic: %f secondes' % (classic_end - classic_start))

    cv2.imwrite('output_cuda.jpg', d_img.copy_to_host())
    cv2.imwrite('output.jpg', img)

if __name__ == '__main__':
    main()