#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512

void fillMatrix(float *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() % 100;
    }
}

void multiply(float *A, float *B, float *C, int size, int rows) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < size; j++) {
            C[i*size + j] = 0;
            for (int k = 0; k < size; k++) {
                C[i*size + j] += A[i*size + k] * B[k*size + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    int rank, size;
    float *A = NULL, *B = NULL, *C = NULL;
    float *local_A, *local_C;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = N / size;

    local_A = (float*)malloc(sizeof(float) * rows * N);
    local_C = (float*)malloc(sizeof(float) * rows * N);
    B = (float*)malloc(sizeof(float) * N * N);

    if (rank == 0) {
        A = (float*)malloc(sizeof(float) * N * N);
        C = (float*)malloc(sizeof(float) * N * N);
        srand(time(0));
        fillMatrix(A, N);
        fillMatrix(B, N);
    }

    double start = MPI_Wtime();

    MPI_Bcast(B, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, rows * N, MPI_FLOAT, local_A, rows * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    multiply(local_A, B, local_C, N, rows);

    MPI_Gather(local_C, rows * N, MPI_FLOAT, C, rows * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("MPI + OpenMP time: %f seconds\n", end - start);
        free(A); free(C);
    }

    free(B); free(local_A); free(local_C);
    MPI_Finalize();
    return 0;
}
