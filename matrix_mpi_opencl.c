#define CL_TARGET_OPENCL_VERSION 210
#include <mpi.h>
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

// Utility to read OpenCL kernel source file
const char* read_kernel(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to load kernel");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);
    char* source = (char*)malloc(size + 1);
    fread(source, 1, size, fp);
    source[size] = '\0';
    fclose(fp);
    return source;
}

int main(int argc, char** argv) {
    const int N = 1024;
    float A[N * N], B[N * N], C[N * N];
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = N / size;

    if (rank == 0) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                A[i * N + j] = i + j;
                B[i * N + j] = i - j;
            }
    }

    float* A_sub = (float*)malloc(rows * N * sizeof(float));
    float* C_sub = (float*)malloc(rows * N * sizeof(float));

    MPI_Bcast(B, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, rows * N, MPI_FLOAT, A_sub, rows * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // --------- OpenCL Setup ---------
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    cl_mem bufA, bufB, bufC;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_queue_properties props[] = { 0 };
    queue = clCreateCommandQueueWithProperties(context, device, props, &err);

    const char* src = read_kernel("matrix_mul.cl");
    program = clCreateProgramWithSource(context, 1, &src, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "matrix_mul", &err);

    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, rows * N * sizeof(float), NULL, &err);
    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &err);
    bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, rows * N * sizeof(float), NULL, &err);

    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, rows * N * sizeof(float), A_sub, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, N * N * sizeof(float), B, 0, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &N);

    size_t global[2] = { rows, N };
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    clFinish(queue);

    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, rows * N * sizeof(float), C_sub, 0, NULL, NULL);

    // Gather final result
    MPI_Gather(C_sub, rows * N, MPI_FLOAT, C, rows * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf(" MPI + OpenCL matrix multiplication complete.\n");
    }

    // Cleanup
    free((void*)src);
    free(A_sub);
    free(C_sub);
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
