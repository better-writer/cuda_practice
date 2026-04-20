#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>

#define TILE_SIZE 32

// 定义 CUDA 错误检查宏
#define CHECK_CUDA(call)                                                 \
    {                                                                    \
        const cudaError_t error = call;                                  \
        if (error != cudaSuccess) {                                      \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);      \
            fprintf(stderr, "code: %d, reason: %s\n", error,             \
                    cudaGetErrorString(error));                          \
            exit(1);                                                     \
        }                                                                \
    }

__global__ void naiveGemm(float *A, float *B, float *C, int M, int N, int K){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = 0.00f;
    if (row < M && col < N) {
        for (int k=0; k < K; k++){
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void tiledGemm(float *A, float *B, float *C, int M, int N, int K){
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.00f;
    int tiles = (K + TILE_SIZE - 1)/TILE_SIZE;

    for (int t = 0; t < tiles; ++t) {
        
        // --- 阶段 A: 加载数据到共享内存 ---
        
        // 1. 加载 A 的块 (行优先)
        // A 的全局行是 row, 全局列是 t * TILE_SIZE + threadIdx.x
        int aRow = row;
        int aCol = t * TILE_SIZE + threadIdx.x;
        
        // 边界检查：防止越界，越界填 0
        if (aRow < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // 2. 加载 B 的块 (行优先)
        // B 的全局行是 t * TILE_SIZE + threadIdx.y, 全局列是 col
        int bRow = t * TILE_SIZE + threadIdx.y;
        int bCol = col;

        // 边界检查
        if (bRow < K && bCol < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        // --- 阶段 B: 同步 ---
        // 必须等待块内所有线程都加载完数据，才能开始计算
        __syncthreads();

        // --- 阶段 C: 计算 ---
        // 从共享内存读取数据计算
        // 注意：这里 As 是按行读，Bs 是按列读（但在共享内存中，Bs 的行其实对应全局 B 的列的一部分）
        // 这里的逻辑是：As[y][k] * Bs[k][x]
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // --- 阶段 D: 再次同步 ---
        // 确保所有线程计算完当前块，才能进入下一轮循环覆盖共享内存
        __syncthreads();
    }

    // --- 4. 写回全局内存 ---
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main(){
    int M = 1<<10, N = 1<<10, K = 1<<10; // 大矩阵尺寸
    size_t sizeA = M * K;
    size_t sizeB = K * N;
    size_t sizeC = M * N;

    float *h_A = (float*)malloc(sizeA * sizeof(float));
    float *h_B = (float*)malloc(sizeB * sizeof(float));
    float *h_C = (float*)malloc(sizeC * sizeof(float));
    float *h_C_naive = (float*)malloc(sizeC * sizeof(float));

    // 初始化 A 和 B
    for (int i = 0; i < M*K; i++) h_A[i] = 1.0f; // 示例数据
    for (int i = 0; i < K*N; i++) h_B[i] = 1.0f;

    

    float *d_A, *d_B, *d_C, *d_C_naive;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_naive, sizeC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    cudaEvent_t start_naive, stop_naive;
    CHECK_CUDA(cudaEventCreate(&start_naive));
    CHECK_CUDA(cudaEventCreate(&stop_naive));
    float seconds_naive = 0;

    dim3 blockSize_naive(16, 16);
    dim3 grid_naive((N + blockSize_naive.x - 1) / blockSize_naive.x, (M + blockSize_naive.y - 1) / blockSize_naive.y);
    naiveGemm<<<grid_naive, blockSize_naive>>>(d_A, d_B, d_C_naive, M, N, K);// 预热 GPU，避免第一次调用的额外开销影响计时
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start_naive));

    naiveGemm<<<grid_naive, blockSize_naive>>>(d_A, d_B, d_C_naive, M, N, K);

    CHECK_CUDA(cudaEventRecord(stop_naive));
    CHECK_CUDA(cudaEventSynchronize(stop_naive));

    CHECK_CUDA(cudaEventElapsedTime(&seconds_naive, start_naive, stop_naive));

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C_naive, d_C_naive, sizeC, cudaMemcpyDeviceToHost));




    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float seconds = 0;

    CHECK_CUDA(cudaEventRecord(start));

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    tiledGemm<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);

    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&seconds, start, stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));


    for (int i = 0; i < M*N; i++) {
        if (h_C[i] != h_C_naive[i]) {
            printf("Mismatch at index %d: tiled %f vs naive %f\n", i, h_C[i], h_C_naive[i]);
            break;
        }
    }
    
    printf("Tiled GEMM Time: %f ms\n", seconds);
    printf("Naive GEMM Time: %f ms\n", seconds_naive);


    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_C_naive));

    return 0;
}



