#include <stdio.h>
#include <stdbool.h>

void saveGolStateToFile(bool *state, const char *path, int size)
{
    FILE *file = fopen(path, "w");
    if (file == NULL)
    {
        printf("Error: Unable to open file %s\n", path);
        return;
    }
    for (int i = 0; i < size * size; ++i)
    {
        if (state[i])
            fprintf(file, "1");
        else
            fprintf(file, "0");
        fprintf(file, ",");
    }
    fclose(file);
    printf("Game of Life state saved to %s\n", path);
}