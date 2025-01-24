#include "game_of_life.h"
#include "save_to_file_util.c"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage: %s <gridSize> <numSteps> <usePinned>\n", argv[0]);
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

    int usePinned = atoi(argv[3]);
    if (usePinned != 0 && usePinned != 1)
    {
        printf("Error: usePinned is a flag and should be either 0 or 1.\n");
        return 1;
    }

    bool *hostState;

    if (usePinned)
        cudaMallocHost(&hostState, size * size * sizeof(bool));
    else
        hostState = (bool *)malloc(size * size * sizeof(bool));

    srand((unsigned)time(NULL));
    for (int i = 0; i < size * size; i++)
        hostState[i] = rand() % 2;

    // KERNEL CALL ///////////////////////////////////////////////

    calculateGameOfLife(hostState, size, numSteps);

    // KERNEL CALL ///////////////////////////////////////////////

    if (usePinned)
        cudaFreeHost(hostState);
    else
        free(hostState);

    printf("Game of Life simulation completed successfully.\n");
    return 0;
}
