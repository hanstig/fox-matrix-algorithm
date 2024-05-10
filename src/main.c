#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

#define N 3

void matrix_mult(double A, double *B, double *C) {
  for (size_t i = 0; i < N * N; ++i) {
    C[i] += A * B[i];
  }
}

int main(int argc, char *argv[]) {

  int rank, size, i, provided;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int dims[] = {N, N};
  int periods[] = {1, 1};

  MPI_Comm new_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &new_comm);

  MPI_Comm_rank(new_comm, &rank);

  int coords[2];
  MPI_Cart_coords(new_comm, rank, 2, coords);

  MPI_Comm row;
  int direction_row[] = {0, 1};
  MPI_Cart_sub(new_comm, direction_row, &row);

  MPI_Comm col;
  int direction_col[] = {1, 0};
  MPI_Cart_sub(new_comm, direction_col, &col);

  double A[N * N];
  double B[N * N];
  double C[N * N];

  double C_val = 0;
  double B_val;

  double A_diag = 0;

  for (int i = 0; i < N * N; ++i) {
    A[i] = i;
    B[i] = i;

    if (rank == i) {
      B_val = i;
    }
  }

  if (rank == 0) {
    printf("A:\n");
    for (int y = 0; y < N; ++y) {
      for (int x = 0; x < N; ++x) {
        printf("%lf ", A[y * N + x]);
      }
      printf("\n");
    }
    printf("\n");

    printf("B:\n");
    for (int y = 0; y < N; ++y) {
      for (int x = 0; x < N; ++x) {
        printf("%lf ", B[y * N + x]);
      }
      printf("\n");
    }
  }

  MPI_Barrier(row);

  int diag = coords[0] * N + coords[0];

  for (size_t i = 0; i < N; ++i) {

    int root = (coords[0] + i) % N;

    MPI_Bcast(coords[1] == root ? &A[N * coords[0] + ((diag + i) % N)]
                                : &A_diag,
              1, MPI_DOUBLE, root, row);

    printf("C[%d %d] += %lf * %lf\n", rank % N, rank / N, A_diag, B_val);

    if (root == coords[1]) {
      A_diag = A[N * coords[0] + ((diag + i) % N)];
    }

    C_val += A_diag * B_val;

    MPI_Sendrecv_replace(&B_val, 1, MPI_DOUBLE, (coords[0] - 1 + N) % N, 0,
                         (coords[0] + 1) % N, 0, col, MPI_STATUS_IGNORE);
  }

  if (rank != 0) {
    MPI_Send(&C_val, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  double C_final[N * N];

  usleep(1000000);
  if (rank == 0) {

    printf("Accumulating\n");

    C_final[0] = C_val;

    for (int i = 1; i < N * N; ++i) {
      // TODO: non blocking recieve and waitgroup
      MPI_Recv(&C_final[i], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    printf("result:\n");
    for (int y = 0; y < N; ++y) {
      for (int x = 0; x < N; ++x) {
        printf("%lf ", C_final[y * N + x]);
      }
      printf("\n");
    }
  }

  MPI_Finalize();
  return 0;
}
