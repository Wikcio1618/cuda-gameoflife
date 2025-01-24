#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

__device__ int countAliveNeis(int idx, bool *state, int size);

__global__ void computeGameOfLifeStep(bool *curr_state, bool *next_state, int size);

void calculateGameOfLife(bool *host_state, int size, int num_steps);

#endif
