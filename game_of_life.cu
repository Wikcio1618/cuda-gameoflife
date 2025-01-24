#include "game_of_life.h"

#include <stdio.h>

__device__ int countAliveNeis(int idx, bool *state, int size)
{
    int result = 0;
    int row = idx / size;
    int col = idx % size;

    // Neighbor offsets for periodic boundary conditions (wrapping grid edges)
    const int neighborOffsets[8][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

    for (int i = 0; i < 8; i++)
    {
        int nRow = (row + neighborOffsets[i][0] + size) % size; // Wrap rows
        int nCol = (col + neighborOffsets[i][1] + size) % size; // Wrap columns
        int neighborIdx = nRow * size + nCol;
        result += state[neighborIdx];
    }

    return result;
}

__global__ void computeGameOfLifeStep(bool *currState, bool *nextState, int size)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < size && col < size)
    {
        int idx = row * size + col;
        int aliveNeis = countAliveNeis(idx, currState, size);
        if (currState[idx] == 0 && aliveNeis == 3)
            nextState[idx] = 1;
        else if (currState[idx] == 1 && (aliveNeis == 2 || aliveNeis == 3))
            nextState[idx] = 1;
        else
            nextState[idx] = 0;
    }
}

void calculateGameOfLife(bool *host_state, int size, int num_steps)
{
    bool *device_state;
    if (cudaMalloc(&device_state, size * size * sizeof(bool)) != cudaSuccess)
    {
        fprintf(stderr, "Error: Failed to allocate memory for CUDA");
    }

    // Copy memory from host to device
    if (cudaMemcpy(device_state, host_state, size * size * sizeof(bool), cudaMemcpyDefault) != cudaSuccess)
    {
        fprintf(stderr, "Error: Failed to copy memory from host to CUDA");
    }

    bool *device_temp_state;
    if (cudaMalloc(&device_temp_state, size * size * sizeof(bool)) != cudaSuccess)
    {
        fprintf(stderr, "Error: Failed to allocate memory for CUDA");
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((size + 15) / 16, (size + 15) / 16);

    for (int i = 0; i < num_steps; ++i)
    {
        computeGameOfLifeStep<<<blocks, threadsPerBlock>>>(device_state, device_temp_state, size);
        cudaDeviceSynchronize();

        bool *temp = device_state;
        device_state = device_temp_state;
        device_temp_state = temp;
    }

    if (cudaMemcpy(host_state, device_state, size * size * sizeof(bool), cudaMemcpyDefault) != cudaSuccess)
    {
        fprintf(stderr, "Error: Failed to copy memory from CUDA to host");
        return;
    }
    
    cudaFree(device_temp_state);
    cudaFree(device_state);
}