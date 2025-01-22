#include "game_of_life.h"

__device__ int countAliveNeis(int idx, bool *golState, int size)
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
        result += golState[neighborIdx];
    }

    return result;
}

__global__ void computeGameOfLifeStep(bool *currGolState, bool *nextGolState, int size)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < size && col < size)
    {
        int idx = row * size + col;
        int aliveNeis = countAliveNeis(idx, currGolState, size);
        if (currGolState[idx] == 0 && aliveNeis == 3)
            nextGolState[idx] = 1;
        else if (currGolState[idx] == 1 && (aliveNeis == 2 || aliveNeis == 3))
            nextGolState[idx] = 1;
        else
            nextGolState[idx] = 0;
    }
}

void calculateGameOfLife(bool *golState, int size, int steps)
{
    bool *tempGolState;
    cudaMalloc(&tempGolState, size * size * sizeof(bool));

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((size + 15) / 16, (size + 15) / 16);

    int count = 0;
    while (count < steps)
    {
        computeGameOfLifeStep<<<blocks, threadsPerBlock>>>(golState, tempGolState, size);
        cudaDeviceSynchronize();

        bool *temp = golState;
        golState = tempGolState;
        tempGolState = temp;

        count++;
    }

    // cudaMemcpy(golState, tempGolState, size * size * sizeof(bool), cudaMemcpyDeviceToHost);
    // cudaFree(tempGolState);
}