#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include "mpi.h"

void show_matrix(int n, int matrix[n][n]){
	int i,j;
	printf ("Matrice finale:\n");
	for (i=0;i<n;i++){
		for (j=0;j<n;j++){
			printf ("%d",matrix[i][j]);
			printf (j<n-1?"\t":"\n");
		}
	}
}

int main (int argc, char *argv[]){
	if (argc == 4){ // check du nombre d'arguments du programme
		double timeStart, timeEnd, Texec;
		struct timeval tp;
		gettimeofday (&tp, NULL); // Debut du chronometre
		timeStart = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
		int err,np,id; 
		MPI_Status status;
		err = MPI_Init(&argc,&argv);
		if (err != MPI_SUCCESS){
			printf("Erreur d'initialisation de MPI\n");
			exit(1);
		}
		MPI_Comm_size(MPI_COMM_WORLD, &np);
		if (np < 17){
			printf("Erreur: le nombre de processeurs doit être de 17 ou plus\n");
		}
		MPI_Comm_rank(MPI_COMM_WORLD, &id);
		int c = atoi(argv[1]);
		int p = atoi(argv[2]);
		int n = atoi(argv[3]);
		int i,j,k;
		int tmp[17];
		if (id < 16){
			tmp[16] = id;
			for (j=0;j<16;j++){
				tmp[j] = p;
			}
			if (c == 1){ // pb 1
				for (k=0;k<=n;k++){
					for (j=0;j<16;j++){
						usleep(1000);
						tmp[j] += (id+2*j)*k;
					}
				}
				MPI_Send(&tmp, 17, MPI_INT, 16, 1, MPI_COMM_WORLD);
			}
			else if (c == 2){
				for (k=0;k<=n;k++){
					for (j=0;j<16;j++){
						usleep(1000);
						if (j == 0){
							tmp[j] += id*k;
						}
						else {
							tmp[j] += tmp[j-1]*k;
						}
					}		
				}
				MPI_Send(&tmp, 17, MPI_INT, 16, 1, MPI_COMM_WORLD);
			}
			else {
				printf ("Erreur: choisir parmi le problème 1 ou 2");
			}
		}
		else {
			int matrix[16][16];
			for (j=0;j<16;j++){
				MPI_Recv(&tmp, 17, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
				for (i=0;i<16;i++){
					matrix[tmp[16]][i] = tmp[i];
				}
			}
			printf ("Routine ID: %d\n", c);
			printf ("Valeur initiale: %d\n", p);
			printf ("Nombre de matrices: %d\n", n);
			show_matrix(16, matrix);
			gettimeofday (&tp, NULL); // Fin du chronometre
			timeEnd = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
			Texec = timeEnd - timeStart;
			printf("Temps d'exécution en secondes = %f\n", Texec);
		}
		MPI_Finalize();
	}
	else {
		printf ("Syntaxe: tp1 c p n");
	}
	return 0;
}
