#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512

void fillMatrix(float *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() % 100;
    }
}

void multiply(float *A, float *B, float *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i*size + j] = 0;
            for (int k = 0; k < size; k++) {
                C[i*size + j] += A[i*size + k] * B[k*size + j];
            }
        }
    }
}

int main() {
    float *A = malloc(sizeof(float) * N * N);
    float *B = malloc(sizeof(float) * N * N);
    float *C = malloc(sizeof(float) * N * N);

    srand(time(0));
    fillMatrix(A, N);
    fillMatrix(B, N);

    clock_t start = clock();
    multiply(A, B, C, N);
    clock_t end = clock();

    printf("Sequential time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(A); free(B); free(C);
    return 0;
}
