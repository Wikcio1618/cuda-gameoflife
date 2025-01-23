#include "game_of_life.h"
#include "save_to_file_util.c"

#include <stdio.h>
#include <stdlib.h>

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

    bool *hostState;

    hostState = (bool *)malloc(size * size * sizeof(bool));
    if (hostState == NULL)
    {
        printf("Error: Failed to allocate memory on the host.\n");
        return 1;
    }

    srand((unsigned)time(NULL));
    for (int i = 0; i < size * size; i++) hostState[i] = rand() % 2;

    // KERNEL CALL ///////////////////////////////////////////////
    
    calculateGameOfLife(hostState, size, numSteps, false);

    // KERNEL CALL ///////////////////////////////////////////////

    free(hostState);

    printf("Game of Life simulation completed successfully.\n");
    return 0;
}
