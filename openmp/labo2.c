#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>

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

float resolve_seq(int c, int p, int n){
	double timeStart, timeEnd, Texec;
	struct timeval tp;
	gettimeofday (&tp, NULL);
	timeStart = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
	int i,j,k=0;
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
					usleep(50000);
					matrix[i][j] += i+2*j;
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
				for (j=15;j>=0;j--){
					usleep(50000);
					if (j == 15){
						matrix[i][j] += 3*i;
					}
					else {
						matrix[i][j] = 2*matrix[i][j] + matrix[i][j+1];
					}
				}		
			}			
		}
		show_matrix(16, matrix);
	}
	else {
		printf ("Erreur: choisir parmi le problème 1 ou 2");
	}
	gettimeofday (&tp, NULL);
	timeEnd = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
	Texec = timeEnd - timeStart;
	return Texec;
}

float resolve_par(int c, int p, int n){
	double timeStart, timeEnd, Texec;
	struct timeval tp;
	gettimeofday (&tp, NULL);
	timeStart = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6; // Temps debut
	int i,j,k=0;
	if (c == 1){ // pb 1
		int matrix[256];
		for (i=0;i<256;i++){
			matrix[i] = p; // Remplissage de la matrice avec la valeur initiale
		}
		printf ("Routine ID: %d\n", c);
		printf ("Valeur initiale: %d\n", p);
		printf ("Nombre de matrices: %d\n", n);
		omp_set_num_threads(omp_get_num_procs()); // Nombre de threads = nombre de processeurs disponibles
		while (k != n){
			k++;
			#pragma omp parallel for // Distribue les iterations de la boucle entre les threads
			for (i=0;i<256;i++){
				usleep(50000);
				matrix[i] += (i/16)+2*(i%16);
			}			
		}
		for (i=0;i<256;i++){
			printf("%d",matrix[i]);
			printf ((i+1)%16!=0?"\t":"\n");
		}
	}
	else if (c == 2){ // pb 2
		int matrix[16][16];
		init_matrix(p, 16, matrix); // Remplissage de la matrice avec la valeur initiale
		printf ("Routine ID: %d\n", c);
		printf ("Valeur initiale: %d\n", p);
		printf ("Nombre de matrices: %d\n", n);
		omp_set_num_threads(16); // Nombre de threads = 16 (nombre de lignes de la matrice)
		while (k != n){
			k++;
			#pragma omp parallel private(j) // Variable correspondant a la colonne de la matrice doit etre privee pour chaque ligne
			{
				#pragma omp for // Distribue les lignes de la matrice entre les threads
				for (i=0;i<16;i++){ 
					for (j=15;j>=0;j--){ // Boucle qui decremente sur les colonnes de la matrice
						usleep(50000);
						if (j == 15){ // Si derniere colonne
							matrix[i][j] += 3*i;
						}
						else {
							matrix[i][j] = 2*matrix[i][j] + matrix[i][j+1];
						}
					}		
				}			
			}
		}
		show_matrix(16, matrix);
	}
	else {
		printf ("Erreur: choisir parmi le problème 1 ou 2");
	}
	gettimeofday (&tp, NULL);
	timeEnd = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
	Texec = timeEnd - timeStart;
	return Texec;
}

int main (int argc, char *argv[]){
	if (argc == 4){
		int c = atoi(argv[1]);
		int p = atoi(argv[2]);
		int n = atoi(argv[3]);
		printf("------------------------------\n\n");
		float texec_seq = resolve_seq(c,p,n);
	       	printf("Temps d'exécution séquentiel: %f\n", texec_seq);	
		printf("\n------------------------------\n\n");
		float texec_par = resolve_par(c,p,n); 
	       	printf("Temps d'exécution parallèle: %f\n", texec_par);
		printf("\n------------------------------\n\n");
		printf("Accélération: %f\n", texec_seq/texec_par);
		printf("\n------------------------------\n\n");
	}
	else {
		printf ("Syntaxe: ./labo2 c p n\n");
		exit(1);
	}
	return 0;
}
