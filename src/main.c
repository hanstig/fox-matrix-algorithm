#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define N 1024

int getIndex(int x, int y);

void matrix_mult(double *A, double *B, double *C, int dim) {
  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++)
      for (int k = 0; k < dim; k++)
        C[i * dim + j] += A[i * dim + k] * B[k * dim + j];
}

void printMatrix(double *m, int dim) {
  for (int y = 0; y < dim; ++y) {
    for (int x = 0; x < dim; ++x) {
      printf("%lf ", m[y * dim + x]);
    }
    printf("\n");
  }
  printf("\n");
}

void getBlockVals(double *matrix, int blockx, int blocky, double *output,
                  int size, int blockSize) {

  int i = blocky * N + blockx;

  int j = 0;
  while (j < N * N / size) {
    for (int x = 0; x < blockSize; ++x) {
      output[j++] = matrix[i++];
    }
    i += N - blockSize;
  }
}

void setBlockVals(double *matrix, int blockx, int blocky, double *input,
                  int size, int blockSize) {

  int i = blocky * N + blockx;

  int j = 0;
  while (j < N * N / size) {
    for (int x = 0; x < blockSize; ++x) {
      matrix[i++] = input[j++];
    }
    i += N - blockSize;
  }
}

int getIndex(int x, int y) { return y * N + x; }

int main(int argc, char *argv[]) {
  int rank, size, i, provided;

  double t1, t2;

  int grid_rank;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int dims[] = {sqrt(size), sqrt(size)};
  int periods[] = {1, 1};
  int coords[2];

  int direction_row[] = {0, 1};
  int direction_col[] = {1, 0};

  // Init matrix
  double *A;
  double *B;
  // double *C;

  // Fill matrices
  if (rank == 0) {

    A = (double *)malloc(N * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));
    // C = (double *)malloc(N * N * sizeof(double));

    for (int i = 0; i < N * N; ++i) {
      A[i] = i;
      B[i] = i;
      // C[i] = 0;
    }
  }

  MPI_Comm grid_comm, row, col;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);
  MPI_Cart_sub(grid_comm, direction_row, &row);
  MPI_Cart_sub(grid_comm, direction_col, &col);
  MPI_Comm_rank(grid_comm, &grid_rank);
  MPI_Cart_coords(grid_comm, rank, 2, coords);

  MPI_Barrier(MPI_COMM_WORLD);

  // Get blocks
  int blockSize = N / (int)sqrt(size);

  // double A_block[blockSize * blockSize];
  // double B_block[blockSize * blockSize];
  // double C_block[blockSize * blockSize];

  double *A_block = (double *)malloc(blockSize * blockSize * sizeof(double));
  double *B_block = (double *)malloc(blockSize * blockSize * sizeof(double));
  double *C_block = (double *)malloc(blockSize * blockSize * sizeof(double));

  t1 = MPI_Wtime();

  // Fill blocks and send to respective ranks
  MPI_Request reqs[2 * size];

  if (rank == 0) {

    getBlockVals(A, 0, 0, A_block, size, blockSize);
    getBlockVals(B, 0, 0, B_block, size, blockSize);
    

    for (int y = 0; y < sqrt(size); y += 1) {
      for (int x = 0; x < sqrt(size); x += 1) {
        if (x == 0 && y == 0)
          continue;

        // double A_temp[blockSize * blockSize];
        // double B_temp[blockSize * blockSize];
        
        double * A_temp = (double *) malloc(blockSize * blockSize * sizeof(double));
        double * B_temp = (double *) malloc(blockSize * blockSize * sizeof(double));

        getBlockVals(A, x * blockSize, y * blockSize, A_temp, size, blockSize);
        getBlockVals(B, x * blockSize, y * blockSize, B_temp, size, blockSize);

        int dst = x + y * (int)sqrt(size);

        MPI_Send(A_temp, blockSize * blockSize, MPI_DOUBLE, dst, 0,
                  MPI_COMM_WORLD);

        MPI_Send(B_temp, blockSize * blockSize, MPI_DOUBLE, dst, 1,
                  MPI_COMM_WORLD);

        free(A_temp);
        free(B_temp);
      }
    }
  } else {

    MPI_Recv(A_block, blockSize * blockSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    MPI_Recv(B_block, blockSize * blockSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  // free(A);
  // free(B);

  // double A_temp[blockSize * blockSize];
  double * A_temp = (double *) malloc(blockSize * blockSize * sizeof(double));

  MPI_Barrier(row);

  int diag = coords[0] * sqrt(size) + coords[0];

  for (size_t i = 0; i < sqrt(size); ++i) {
    int root = (coords[0] + i) % (int)sqrt(size);

    MPI_Bcast(coords[1] == root ? A_block : A_temp, blockSize * blockSize,
              MPI_DOUBLE, root, row);

    if (root == coords[1]) {
      memcpy(A_temp, A_block, blockSize * blockSize * sizeof(double));
    }

    matrix_mult(A_temp, B_block, C_block, blockSize);

    MPI_Sendrecv_replace(B_block, blockSize * blockSize, MPI_DOUBLE,
                         (coords[0] - 1 + (int)sqrt(size)) % (int)sqrt(size), 0,
                         (coords[0] + 1) % (int)sqrt(size), 0, col,
                         MPI_STATUS_IGNORE);
  }

  if (rank != 0) {
    MPI_Send(C_block, blockSize * blockSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    free(A_block);
    free(C_block);
    free(B_block);
    free(A_temp);
  }

  // double C_final[N * N];

  if (rank == 0) {

    double * C_final = (double*)malloc(N * N * sizeof(double));
    setBlockVals(C_final, 0, 0, C_block, size, blockSize);

    free(A_block);
    free(C_block);
    free(B_block);
    free(A_temp);

    for (int y = 0; y < (int)sqrt(size); y += 1) {
      for (int x = 0; x < (int)sqrt(size); x += 1) {
        if (x == 0 && y == 0)
          continue;

        double C_temp[blockSize * blockSize];
        MPI_Recv(&C_temp, blockSize * blockSize, MPI_DOUBLE,
                 y * (int)sqrt(size) + x, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        setBlockVals(C_final, x * blockSize, y * blockSize, C_temp, size,
                     blockSize);
      }
    }

    t2 = MPI_Wtime();

    // printf("result:\n");
    // for (int y = 0; y < N; ++y) {
    //   for (int x = 0; x < N; ++x) {
    //     printf("%lf ", C_final[y * N + x]);
    //   }
    //   printf("\n");
    // }

    printf("time: %lf seconds\n", t2 - t1);
    
    free(C_final);
    // free(A);
    // free(B);
    // free(C);
  }

  MPI_Finalize();
  return 0;
}
