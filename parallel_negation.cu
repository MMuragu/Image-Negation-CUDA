#include <stdio.h>
#include "image.h"

__global__ void negative_kernel(unsigned char *pixel,
                                unsigned char max_value, int n) {
    /* calculate thread ID */
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    /* calculate number of threads from grid dimension */
    int num_threads = gridDim.x * blockDim.x;
    
    /* use a while loop to process data, jumping appropriate index */
    while (tid < n) {
        pixel[tid] = max_value - pixel[tid];
        tid += num_threads;
    }
}

void process_data(image *photo) {
    int n = photo->width * photo->height * 3;
    unsigned char *d_pixel;

    /* step 1: allocate device memory */
    cudaMalloc((void **)&d_pixel, n);

    /* step 2: copy pixel data to device memory */
    cudaMemcpy(d_pixel, photo->data, n, cudaMemcpyHostToDevice);

    /* step 3: invoke kernel function */
    negative_kernel<<<256, 256>>>(d_pixel, (unsigned char)photo->max_value, n);
    cudaDeviceSynchronize();

    /* step 4: copy result from device memory to host */
    cudaMemcpy(photo->data, d_pixel, n, cudaMemcpyDeviceToHost);

    /* step 5: clear device memory */
    cudaFree(d_pixel);
}

image *setup(int argc, char **argv) {
    image *photo;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <infile> <outfile>\n\n", argv[0]);
        return NULL;
    }

    photo = read_image(argv[1]);
    if (photo == NULL) {
        fprintf(stderr, "Unable to read input file %s\n\n", argv[1]);
        return NULL;
    }

    return photo;
}

void cleanup(image *photo, char **argv) {
    int rc = write_image(argv[2], photo);
    if (!rc) {
        fprintf(stderr, "Unable to write output file %s\n\n", argv[2]);
    }

    clear_image(photo);
}

int main(int argc, char **argv) {
    image *photo = setup(argc, argv);
    if (photo != NULL) {
        process_data(photo);
        cleanup(photo, argv);
    }
    return 0;
}

