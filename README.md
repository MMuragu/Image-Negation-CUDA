# CUDA Image Negation - Parallel Processing Solution

CUDA image negation solver - Fixes Tesla T4 PTX compilation issues in Colab. 

## Project Structure

* `parallel_negation.cu` - CUDA parallel implementation using 256x256 thread grid
* `sequential_negation.cu` - Reference CPU implementation for verification
* `image.h` - Image data structure and function declarations
* `image.cu` - Image I/O functions for PPM P6 format
* `example.ppm` - Sample 640x426 test image

## Prerequisites

* Google Colab account with GPU runtime
* Basic understanding of CUDA programming
* PPM image format knowledge

## Installation & Execution

### 1. Setup Colab Environment

```bash
# Switch to GPU runtime: Runtime -> Change runtime type -> GPU
# Verify CUDA availability
!nvidia-smi
```

### 2. Upload Files to Colab

Upload all project files to your Colab environment:

* `parallel_negation.cu`
* `sequential_negation.cu`
* `image.h`
* `image.cu`
* `example.ppm`

### 3. Compile the Programs

```bash
# Compile sequential version
!nvcc -o sequential_negation sequential_negation.cu image.cu

# Compile parallel version with Tesla T4 optimization
!nvcc -arch=sm_70 -o parallel_negation parallel_negation.cu image.cu
```

### 4. Execute Image Processing

```bash
# Run sequential processing
!./sequential_negation example.ppm seq_example.ppm

# Run parallel processing  
!./parallel_negation example.ppm par_example.ppm
```

### 5. Verify Results

```bash
# Check if outputs are identical
!diff seq_example.ppm par_example.ppm

# No output from diff indicates successful parallel implementation
```

## Key Implementation Details

### Parallel Kernel Design

```cpp
__global__ void negative_kernel(unsigned char *pixel,
                                unsigned char max_value, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = gridDim.x * blockDim.x;
    
    while (tid < n) {
        pixel[tid] = max_value - pixel[tid];
        tid += num_threads;
    }
}
```

* **Grid Configuration:** 256 blocks with 256 threads each (65,536 total threads)
* **Dynamic stride-based processing** for arbitrary image sizes
* **Proper memory management** with `cudaMalloc` / `cudaMemcpy`

### Common Issues Resolved

* **PTX Compilation Error**

  * Problem: "the provided PTX was compiled with an unsupported toolchain"
  * Solution: Use `-arch=sm_70` flag for Tesla T4 compatibility

* **Kernel Not Executing**

  * Problem: CUDA kernels compile but don't modify data
  * Solution: Ensure proper `cudaDeviceSynchronize()` and error checking

* **PPM Format Issues**

  * Problem: Incorrect image reading/writing
  * Solution: Use provided `image.cu` functions for PPM P6 format

### Performance Comparison

* **Sequential:** Processes pixels one-by-one on CPU
* **Parallel:** Processes multiple pixels simultaneously on GPU
* **Verification:** Identical output ensures correctness


## Viewing Results

Since Colab doesn't natively support PPM viewing, convert to PNG:

```bash
# Install conversion tools
!apt-get install netpbm

# Convert to viewable format
!ppmtopng par_example.ppm > result.png

# Download for local viewing
from google.colab import files
files.download('result.png')
```

