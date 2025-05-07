// matrix_mpi_opencl.c
#include <mpi.h>
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512
#define CL_CHECK(err) if (err != CL_SUCCESS) { printf("OpenCL error %d\n", err); exit(1); }

const char *kernelSource = 
"__kernel void mat_mul(__global float* A, __global float* B, __global float* C, int N, int rows) {\n"
"  int i = get_global_id(0);\n"
"  int j = get_global_id(1);\n"
"  if (i < rows && j < N) {\n"
"    float sum = 0.0f;\n"
"    for (int k = 0; k < N; ++k) {\n"
"      sum += A[i*N + k] * B[k*N + j];\n"
"    }\n"
"    C[i*N + j] = sum;\n"
"  }\n"
"}\n";

void fillMatrix(float *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() % 100;
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

    // OpenCL Setup
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem bufA, bufB, bufC;
    cl_int err;

    CL_CHECK(clGetPlatformIDs(1, &platform, NULL));
    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL));
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CL_CHECK(err);
    queue = clCreateCommandQueue(context, device, 0, &err); CL_CHECK(err);

    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * rows * N, local_A, &err); CL_CHECK(err);
    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * N, B, &err); CL_CHECK(err);
    bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * rows * N, NULL, &err); CL_CHECK(err);

    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err); CL_CHECK(err);
    CL_CHECK(clBuildProgram(program, 1, &device, NULL, NULL, NULL));
    kernel = clCreateKernel(program, "mat_mul", &err); CL_CHECK(err);

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &rows));

    size_t global[2] = {rows, N};
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL));
    CL_CHECK(clFinish(queue));

    CL_CHECK(clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * rows * N, local_C, 0, NULL, NULL));

    MPI_Gather(local_C, rows * N, MPI_FLOAT, C, rows * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("MPI + OpenCL time: %f seconds\n", end - start);
        free(A); free(C);
    }

    free(B); free(local_A); free(local_C);

    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    MPI_Finalize();
    return 0;
}
