#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi/mpi.h>

#define DEBUG 1

/* Translation of the DNA bases
   A -> 0
   C -> 1
   G -> 2
   T -> 3
   N -> 4*/

#define M  1000000 // Number of sequences
#define N  200 // Number of bases per sequence

unsigned int g_seed = 0;

int fast_rand(void) {
    g_seed = (214013*g_seed+2531011);
    return (g_seed>>16) % 5;
}

// The distance between two bases
int base_distance(int base1, int base2){

    if((base1 == 4) || (base2 == 4)){
        return 3;
    }

    if(base1 == base2) {
        return 0;
    }

    if((base1 == 0) && (base2 == 3)) {
        return 1;
    }

    if((base2 == 0) && (base1 == 3)) {
        return 1;
    }

    if((base1 == 1) && (base2 == 2)) {
        return 1;
    }

    if((base2 == 2) && (base1 == 1)) {
        return 1;
    }

    return 2;
}

int main(int argc, char *argv[] ) {

    int i, j;
    int *data1, *data2, *data1local, *data2local, *resultlocal;
    int *result;
    struct timeval  tv1, tv2, tv0, tv3;
    int numprocs, rank, block;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    block=M/numprocs;

    if(M%numprocs){
        block++;
    }

    /* Initialize Matrices */
    if(rank==0){
        data1 = (int *) malloc(numprocs*block*N*sizeof(int));
        data2 = (int *) malloc(numprocs*block*N*sizeof(int));
        result = (int *) malloc(numprocs*block*sizeof(int));
        for(i=0;i<M;i++) {
            for(j=0;j<N;j++) {
                /* random with 20% gap proportion */
                data1[i*N+j] = fast_rand();
                data2[i*N+j] = fast_rand();
            }
        }
    }

    data1local= (int *) malloc(block*N*sizeof(int));
    data2local= (int *) malloc(block*N*sizeof(int));
    resultlocal= (int *) malloc(block*sizeof(int));

    gettimeofday(&tv0, NULL);

    MPI_Scatter(data1, block*N, MPI_INT, data1local, block*N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(data2, block*N, MPI_INT, data2local, block*N, MPI_INT, 0, MPI_COMM_WORLD);

    gettimeofday(&tv1, NULL);

    for(i=0;i<block;i++) {
        resultlocal[i]=0;
        for(j=0;j<N;j++) {
            resultlocal[i] += base_distance(data1local[i*N+j], data2local[i*N+j]);
        }
    }

    gettimeofday(&tv2, NULL);

    MPI_Gather(resultlocal, block, MPI_INT, result, block, MPI_INT, 0, MPI_COMM_WORLD);

    gettimeofday(&tv3, NULL);

    int microseconds= (tv2.tv_usec - tv1.tv_usec)+ 1000000 * (tv2.tv_sec - tv1.tv_sec);
    int microseconds2= (tv3.tv_usec - tv0.tv_usec)+ 1000000 * (tv3.tv_sec - tv0.tv_sec);
    microseconds2=microseconds2-microseconds;

    /* Display result */
    if(DEBUG!=0) {
        if(rank==0) {
            if (DEBUG == 1) {
                int checksum = 0;
                for (i = 0; i < M; i++) {
                    checksum += result[i];
                }
                printf("Checksum: %d\n ", checksum);
            } else if (DEBUG == 2) {
                for (i = 0; i < M; i++) {
                    printf(" %d \t ", result[i]);
                }
            }
        }
    } else{
        printf ("Proceso %d: Tiempo de computación= %lf   Tiempo de comunicaciones= %lf\n\n",rank, microseconds/1E6, microseconds2/1E6);
    }

    if(rank==0){
        free(data1); free(data2); free(result);
    }
    free(data1local); free(data2local); free(resultlocal);
    MPI_Finalize();

    return 0;
}
