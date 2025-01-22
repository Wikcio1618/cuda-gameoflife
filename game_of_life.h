#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

__device__ int countAliveNeis(int idx, bool *golState, int size);

__global__ void computeGameOfLifeStep(bool *currentGolState, bool *nextGolState, int size);

void calculateGameOfLife(bool *golState, int size, int steps);

#endif
