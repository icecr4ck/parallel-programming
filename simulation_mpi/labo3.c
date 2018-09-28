#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include "mpi.h"

#define TEMPS_ATTENTE 5

MPI_Datatype init_worker_mpi;

struct init_worker {
	int nb_right;
	int nb_left;
	int num_col;
	int nb_col;
};

void init_matrix(int m, int n, double matrix[2][m][n]){
	int i,j;
	for (i=0;i<m;i++){
		for (j=0;j<n;j++){
			matrix[0][i][j] = 0;
			matrix[1][i][j] = (double) i*(m-i-1)*j*(n-j-1);
		}
	}
}

void show_matrix(int fini, int m, int n, double matrix[m][n]){
	int i,j;
	char *msg = fini ? "Matrice finale:\n":"Matrice initiale:\n";
	printf("%s",msg);
	for (i=0;i<m;i++){
		for (j=0;j<n;j++){
			printf("%.1f\t",matrix[i][j]);
		}
		printf("\n");
	}
}

float resolve_seq(int m, int n, int np, float td, float h){
	double timeStart, timeEnd, Texec;
	struct timeval tp;
	gettimeofday (&tp, NULL);
	timeStart = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
	int i,j,k,l=0;
	double matrix[2][m][n];
	init_matrix(m,n,matrix);
	show_matrix(0,m,n,matrix[1]);
	for (k=1;k<=np;k++){
		for (i=1;i<m-1;i++){
			for (j=1;j<n-1;j++){
				usleep(TEMPS_ATTENTE);
				matrix[l][i][j] = (1-4*td/(h*h)) * matrix[1-l][i][j] + (td/(h*h)) * (matrix[1-l][i-1][j]+matrix[1-l][i+1][j]+matrix[1-l][i][j-1]+matrix[1-l][i][j+1]);
			}
		}
		l=k%2;
	}
	show_matrix(1,m,n,matrix[1-np%2]);
	gettimeofday (&tp, NULL);
	timeEnd = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
	Texec = timeEnd - timeStart;
	return Texec;
}

float resolve_par(int m, int n, int np, float td, float h, int nbproc, int rank){
	int i;
	double matrix[2][m][n];
	MPI_Status status;
	init_matrix(m,n,matrix);
	if (rank == 0){
		double timeStart, timeEnd, Texec;
		struct timeval tp;
		gettimeofday (&tp, NULL);
		timeStart = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
		show_matrix(0,m,n,matrix[1]);
		int nb_col, num_col = 1;
		int nb_col_init = (m-2)/(nbproc-1); // Nombre de colonnes par worker
		int nb_col_reste = (m-2)%(nbproc-1); // Si le nombre de colonnes n'est pas un multiple du nombre de workers
		struct init_worker iw_array[nbproc-1]; // Tableau contenant les infos initiales des workers
		// Envoi des infos initiales aux workers
		for (i=1;i<nbproc;i++){
			nb_col = (i <= nb_col_reste) ? nb_col_init+1 : nb_col_init; // Attribue le bon nombre de colonnes selon le reste
			iw_array[i].nb_right = i+1;
			iw_array[i].nb_left = i-1;
			iw_array[i].num_col = num_col;
			iw_array[i].nb_col = nb_col;
			MPI_Send(&iw_array[i], 1, init_worker_mpi, i, 1, MPI_COMM_WORLD);
			num_col += nb_col;
		}
		// Reception des resultats des workers
		for (i=1;i<nbproc;i++){
			MPI_Recv(&matrix[1-np%2][iw_array[i].num_col][0], iw_array[i].nb_col*n, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
		}
		show_matrix(1,m,n,matrix[1-np%2]);
		gettimeofday (&tp, NULL);
		timeEnd = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
		Texec = timeEnd - timeStart;
		return Texec;
	}
	else {
		int j,k,l=0;
		struct init_worker iw;
		if (rank < nbproc){
			MPI_Recv(&iw, 1, init_worker_mpi, 0, 1, MPI_COMM_WORLD, &status);
			if (iw.nb_col != 0){
				for (k=1;k<=np+1;k++){
					// Envoi et reception du voisin de gauche
					if (iw.nb_left != 0){
						MPI_Send(&matrix[1-l][iw.num_col][0], n, MPI_DOUBLE, iw.nb_left, 1, MPI_COMM_WORLD);
						MPI_Recv(&matrix[1-l][iw.num_col-1][0], n, MPI_DOUBLE, iw.nb_left, 1, MPI_COMM_WORLD, &status);
					}
					//Envoi et reception du voisin de droite
					if (iw.nb_right < m-1 && iw.nb_right < nbproc){
						MPI_Send(&matrix[1-l][iw.num_col+iw.nb_col-1][0], n, MPI_DOUBLE, iw.nb_right, 1, MPI_COMM_WORLD);
						MPI_Recv(&matrix[1-l][iw.num_col+iw.nb_col][0], n, MPI_DOUBLE, iw.nb_right, 1, MPI_COMM_WORLD, &status);
					}
					for (i=iw.num_col;i<(iw.num_col+iw.nb_col);i++){
						for (j=1;j<n-1;j++){
							usleep(TEMPS_ATTENTE);
							matrix[l][i][j] = (1-4*td/(h*h)) * matrix[1-l][i][j] + (td/(h*h)) * (matrix[1-l][i-1][j]+matrix[1-l][i+1][j]+matrix[1-l][i][j-1]+matrix[1-l][i][j+1]);
						}
					}
					l=k%2;
				}
			}
			MPI_Send(&matrix[l][iw.num_col][0], iw.nb_col*n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
		}
		return 0;
	}
}

void add_worker_struct_to_mpi(){
	int blocklen[] = {1, 1, 1, 1};
	MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};
	MPI_Aint offsets[] = {offsetof(struct init_worker, nb_right), offsetof(struct init_worker, nb_left), offsetof(struct init_worker, num_col), offsetof(struct init_worker, nb_col)};
	MPI_Type_create_struct(4, blocklen, offsets, types, &init_worker_mpi);
	MPI_Type_commit(&init_worker_mpi);
}


int main (int argc, char *argv[]){
	if (argc == 7){
		int n = atoi(argv[1]);
		int m = atoi(argv[2]);
		int np = atoi(argv[3]);
		float td = atof(argv[4]);
		float h = atof(argv[5]);
		int nbproc = atoi(argv[6]);
		int nbproc_real, rank;
		int err = MPI_Init(&argc, &argv);
		if (err != MPI_SUCCESS){
			if (rank == 0) printf("Erreur d'initialisation de MPI\n");
			exit(1);
                }
		MPI_Comm_size(MPI_COMM_WORLD, &nbproc_real);
		if (nbproc > nbproc_real){
			if (rank == 0) printf("Le nombre de processeurs ne correspond pas.\n");
			exit(1);
		}
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		add_worker_struct_to_mpi();
		if (rank == 0){
			printf("------------------------------\n");
			printf("Version séquentielle\n");
			printf("------------------------------\n\n");
			float texec_seq = resolve_seq(m,n,np,td,h);
			printf("Temps d'exécution séquentiel: %f\n", texec_seq);	
			printf("\n------------------------------\n");
			printf("Version parallèle\n");
			printf("------------------------------\n\n");
			float texec_par = resolve_par(m,n,np,td,h,nbproc,rank); 
			printf("Temps d'exécution parallèle: %f\n", texec_par);
			printf("\n------------------------------\n");
			printf("Accélération: %f\n", texec_seq/texec_par);
			printf("------------------------------\n\n");
		}
		else {
			resolve_par(m,n,np,td,h,nbproc,rank);
		}
		MPI_Finalize();
	}
	else {
		printf ("Syntaxe: ./labo3 n m np td h nbproc\n");
		exit(1);
	}
	return 0;
}
