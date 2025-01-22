#include "game_of_life.h"

#include <stdio.h>
#include <stdlib.h>

void saveGolStateToFile(bool *golState, const char *path, int size)
{
    FILE *file = fopen(path, "w");
    if (file == NULL)
    {
        printf("Error: Unable to open file %s\n", path);
        return;
    }
    for (int i = 0; i < size * size; ++i)
    {
        if (golState[i])
            fprintf(file, "1");
        else
            fprintf(file, "0");
        fprintf(file, ",");
    }
    fclose(file);
    printf("Game of Life state saved to %s\n", path);
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s <grid_size> <num_steps>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    if (size <= 0)
    {
        printf("Error: Grid size must be a positive integer.\n");
        return 1;
    }

    int numSteps = atoi(argv[2]);
    if (numSteps <= 0)
    {
        printf("Error: Number of steps must be a positive integer.\n");
        return 1;
    }

    bool *golState;
    bool *d_golState; // Device pointer

    golState = (bool *)malloc(size * size * sizeof(bool));
    if (golState == NULL)
    {
        printf("Error: Failed to allocate memory on the host.\n");
        return 1;
    }

    srand((unsigned)time(NULL));
    for (int i = 0; i < size * size; i++) golState[i] = rand() % 2;


    if (cudaMalloc((void **)&d_golState, size * size * sizeof(bool)) != cudaSuccess)
    {
        printf("Error: Failed to allocate memory on the GPU.\n");
        free(golState);
        return 1;
    }

    if (cudaMemcpy(d_golState, golState, size * size * sizeof(bool), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("Error: Failed to copy memory from host to device.\n");
        cudaFree(d_golState);
        free(golState);
        return 1;
    }

    saveGolStateToFile(golState, "pre.txt", size);

    // KERNEL CALL ///////////////////////////////////////////////
    
    calculateGameOfLife(d_golState, size, numSteps);

    // KERNEL CALL ///////////////////////////////////////////////


    if (cudaMemcpy(golState, d_golState, size * size * sizeof(bool), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        printf("Error: Failed to copy memory from device to host.\n");
        cudaFree(d_golState);
        free(golState);
        return 1;
    }

    saveGolStateToFile(golState, "post.txt", size);

    cudaFree(d_golState);
    free(golState);

    printf("Game of Life simulation completed successfully.\n");
    return 0;
}
