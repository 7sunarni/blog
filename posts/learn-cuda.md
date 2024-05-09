# Learn CUDA

Merge Sort using cuda
```cuda
// cat merge_sort.cu
// nvcc merge_sort.cu -o merge_sort
// ./merge_sort
#include <stdio.h>
#include <stdlib.h>
#define N 256 // Sort array size

__global__ void merge(int step, float *array, float *placeholder)
{
    /*
    merge two sorted slice s1 s2
    s1Begin means slice1 begin index
    s1End means slice1 end index
    s2Begin and s2End also for slice2
    */
    int s1Begin = threadIdx.x * (step + 1);
    int s2End = s1Begin + step;
    int s1End = (s1Begin + s2End) / 2;
    int s2Begin = s1End + 1;
    int s1Cursor = s1Begin;
    int s2Cursor = s2Begin;
    int cursor = s1Begin;

    for (;;)
    {
        if (cursor > s2End)
        {
            break;
        }
        if (s1Cursor > s1End)
        {
            placeholder[cursor] = array[s2Cursor];
            s2Cursor = s2Cursor + 1;
            cursor = cursor + 1;
            continue;
        }
        if (s2Cursor > s2End)
        {
            placeholder[cursor] = array[s1Cursor];
            s1Cursor = s1Cursor + 1;
            cursor = cursor + 1;
            continue;
        }
        if (array[s1Cursor] < array[s2Cursor])
        {
            placeholder[cursor] = array[s1Cursor];
            s1Cursor = s1Cursor + 1;
            cursor = cursor + 1;
        }
        else
        {
            placeholder[cursor] = array[s2Cursor];
            s2Cursor = s2Cursor + 1;
            cursor = cursor + 1;
        }
    }
    for (int i = s1Begin; i <= s2End; i++)
    {
        array[i] = placeholder[i];
    }
}

int main()
{
    // TODO(@7sunarni): remove variable placeHolder
    // I'm poor with C programming,
    // I don't know how to init a dynamic array to store unsorted data in merge function, so I use placeHolder to store :<
    float *hostArray, *deviceArray, *placeHolder;
    hostArray = (float *)malloc(sizeof(float) * N);
    printf("before sort \n");
    for (int i = 0; i < N; i++)
    {
        hostArray[i] = (float)rand() / RAND_MAX;
        printf("%f \n", hostArray[i]);
    }
    cudaMalloc(&deviceArray, sizeof(float) * N);
    cudaMalloc(&placeHolder, sizeof(float) * N);
    cudaMemcpy(deviceArray, hostArray, N * sizeof(float), cudaMemcpyHostToDevice);

    int merge_size = 1;
    for (;;)
    {
        int threads = 1 << merge_size;
        if (threads > N)
        {
            break;
        }
        merge<<<1, N / threads>>>(threads - 1, deviceArray, placeHolder);
        merge_size = merge_size + 1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(hostArray, deviceArray, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("after sort\n");
    for (int i = 1; i < N; i++)
    {
        flout prev = hostArray[i-1];
        if (hostArray[i] < prev){
            printf("Failed: %d %f small than previos %f", i, hostArray[i], prev);
            break;
        }
        printf("%f \n", hostArray[i]);
    }
    cudaFree(deviceArray);
    cudaFree(placeHolder);
    return 0;
}

```
