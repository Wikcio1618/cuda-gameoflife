#include "game_of_life.h"
#include "save_to_file_util.c"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage: %s <grid_size> <num_steps> <use_pinned>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    if (size <= 0)
    {
        printf("Error: Grid size must be a positive integer.\n");
        return 1;
    }

    int num_steps = atoi(argv[2]);
    if (num_steps <= 0)
    {
        printf("Error: Number of steps must be a positive integer.\n");
        return 1;
    }

    int use_pinned = atoi(argv[3]);
    if (use_pinned != 0 && use_pinned != 1)
    {
        printf("Error: use_pinned is a flag and should be either 0 or 1.\n");
        return 1;
    }

    bool *host_state;

    if (use_pinned)
        cudaMallocHost(&host_state, size * size * sizeof(bool));
    else
        host_state = (bool *)malloc(size * size * sizeof(bool));

    srand((unsigned)time(NULL));
    for (int i = 0; i < size * size; i++)
        host_state[i] = rand() % 2;

    // KERNEL CALL ///////////////////////////////////////////////

    calculateGameOfLife(host_state, size, num_steps);

    // KERNEL CALL ///////////////////////////////////////////////

    if (use_pinned)
        cudaFreeHost(host_state);
    else
        free(host_state);

    printf("Game of Life simulation completed successfully.\n");
    return 0;
}
