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

void calculateGameOfLife(bool *hostState, int size, int steps, bool usePinned)
{
    bool *deviceState;
    // Allocate memory according to usePinned flag
    if (usePinned)
    {
        if (cudaMallocHost(&deviceState, size * size * sizeof(bool)) != cudaSuccess)
        {
            fprintf(stderr, "Error: Failed to allocate pinned memory for CUDA");
        }
    }
    else if (cudaMalloc(&deviceState, size * size * sizeof(bool)) != cudaSuccess)
    {
        fprintf(stderr, "Error: Failed to allocate memory for CUDA");
    }

    // Copy memory from host to device
    if (cudaMemcpy(deviceState, hostState, size * size * sizeof(bool), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        fprintf(stderr, "Error: Failed to copy memory from host to CUDA");
    }

    bool *deviceTempState;
    // Allocate memory according to usePinned flag
    if (usePinned)
    {
        if (cudaMallocHost(&deviceTempState, size * size * sizeof(bool)) != cudaSuccess)
        {
            fprintf(stderr, "Error: Failed to allocate pinned memory for CUDA");
        }
    }
    else if (cudaMalloc(&deviceTempState, size * size * sizeof(bool)) != cudaSuccess)
    {
        fprintf(stderr, "Error: Failed to allocate memory for CUDA");
    }

    // Copy memory from host to device
    if (cudaMemset(deviceTempState, 0, size * size * sizeof(bool)) != cudaSuccess)
    {
        fprintf(stderr, "Error: Failed to set memory for CUDA");
        return;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((size + 15) / 16, (size + 15) / 16);

    for (int i = 0; i < steps; ++i)
    {
        computeGameOfLifeStep<<<blocks, threadsPerBlock>>>(deviceState, deviceTempState, size);
        cudaDeviceSynchronize();

        bool *temp = deviceState;
        deviceState = deviceTempState;
        deviceTempState = temp;
    }

    if (cudaMemcpy(hostState, deviceState, size * size * sizeof(bool), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        fprintf(stderr, "Error: Failed to copy memory from CUDA to host");
        return;
    }

    if (usePinned)
        cudaFreeHost(deviceTempState);
    else
        cudaFree(deviceTempState);
    if (usePinned)
        cudaFreeHost(deviceState);
    else
        cudaFree(deviceState);
}