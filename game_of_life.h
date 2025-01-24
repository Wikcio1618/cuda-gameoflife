#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

__device__ int countAliveNeis(int idx, bool *state, int size);

__global__ void computeGameOfLifeStep(bool *currState, bool *nextState, int size);

void calculateGameOfLife(bool *hostState, int size, int steps);

#endif
