#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include<stdio.h>
#include<vector>
#include<cmath>
#include<iostream>
#include<chrono>
using namespace std;

// 编写一个CUDA内核函数，执行向量加法：C = A + B
__global__ void vector_add(float *a, float *b, float *c, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N){
        c[idx] = a[idx] + b[idx];
    }
}

//核函数声明
__global__ void vector_add(float *a, float *b, float *c, int N);

void vectorAddCPU(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(){
    int N = 1<<20;
    vector<float> a(N), b(N), c(N), c_cpu(N);

    //生成输入数据a和b
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }

    // 在GPU上分配内存并将数据从主机复制到设备
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // 创建CUDA事件以测量GPU计算时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 将数据从主机复制到设备
    cudaMemcpy(d_a, a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);  //计时开始

    int num_thread = 256;
    int num_block = (N + num_thread - 1)/num_thread;

    vector_add<<<num_block, num_thread>>>(d_a, d_b, d_c, N);

    cudaEventRecord(stop);  //计时结束
    
    float seconds = 0;
    //计算GPU计算时间
    cudaEventElapsedTime(&seconds, start, stop);

    //同步以确保GPU计算完成
    cudaDeviceSynchronize();

    //将结果从设备复制回主机
    cudaMemcpy(c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    
    auto start_cpu = chrono::high_resolution_clock::now();  //cpu计算计时开始
    vectorAddCPU(a.data(), b.data(), c_cpu.data(), N);
    auto stop_cpu = chrono::high_resolution_clock::now();   //cpu计算计时结束
    auto duration_cpu = chrono::duration_cast<chrono::microseconds>(stop_cpu - start_cpu);  //计算CPU计算时间
    double cpu_time_ms = duration_cpu.count() / 1000.0; // 转换为毫秒

    bool passed = true;
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        float error = fabs(c[i] - c_cpu[i]);
        if (error > maxError) maxError = error;
        // 浮点数比较通常允许极小的误差 (如 1e-5)
        if (error > 1e-5) {
            passed = false;
            printf("错误: 索引 %d, GPU结果: %f, CPU结果: %f\n", i, c[i], c_cpu[i]);
            break; // 发现错误就停止
        }
    }

    // 输出测试结果和性能数据
    if (passed) {
        printf("测试通过! 最大误差: %f\n", maxError);
    } else {
        printf("测试失败! 最大误差: %f\n", maxError);
    }
    printf("GPU计算时间: %f ms\n", seconds);
    printf("CPU计算时间: %f ms\n", cpu_time_ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 释放GPU内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

