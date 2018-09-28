#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

void init_matrix(int val, int n, int matrix[n][n]){
	int i,j;
	for (i=0;i<n;i++){
		for (j=0;j<n;j++){
			matrix[i][j] = val;
		}
	}
}

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
		int i,j,k=0;
		int c = atoi(argv[1]);
		int p = atoi(argv[2]);
		int n = atoi(argv[3]);
		int matrix[16][16];
		init_matrix(p, 16, matrix);
		if (c == 1){ // pb 1
			printf ("Routine ID: %d\n", c);
			printf ("Valeur initiale: %d\n", p);
			printf ("Nombre de matrices: %d\n", n);
			while (k != n){
				k++;
				for (i=0;i<16;i++){
					for (j=0;j<16;j++){
						usleep(1000);
						matrix[i][j] += (i+2*j)*k;
					}		
				}			
			}
			show_matrix(16, matrix);
		}
		else if (c == 2){ // pb 2
			printf ("Routine ID: %d\n", c);
			printf ("Valeur initiale: %d\n", p);
			printf ("Nombre de matrices: %d\n", n);
			while (k != n){
				k++;
				for (i=0;i<16;i++){
					for (j=0;j<16;j++){
						usleep(1000);
						if (j == 0){
							matrix[i][j] += i*k;
						}
						else {
							matrix[i][j] += matrix[i][j-1]*k;
						}
					}		
				}			
			}
			show_matrix(16, matrix);
		}
		else {
			printf ("Erreur: choisir parmi le problème 1 ou 2");
		}
		gettimeofday (&tp, NULL); // Fin du chronometre
		timeEnd = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
		Texec = timeEnd - timeStart; //Temps d'execution en secondes
		printf("Temps d'exécution en secondes = %f\n", Texec);
	}
	else {
		printf ("Syntaxe: tp1 c p n");
	}
	return 0;
}
