#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <OpenCL/opencl.h>

void init_matrix(int m, int n, float matrix[2][m][n]){
  int i,j;
  for (i=0;i<m;i++){
    for (j=0;j<n;j++){
      matrix[0][i][j] = (float) i*(m-i-1)*j*(n-j-1);
      matrix[1][i][j] = matrix[0][i][j];
    }
  }
}

void show_matrix(int fini, int m, int n, float matrix[m][n]){
  int i,j;
  char *msg = fini ? "Matrice finale:\n":"Matrice initiale:\n";
  printf("%s",msg);
  for (j=0;j<n;j++){
   for (i=0;i<m;i++){
    printf("%.1f\t",matrix[i][j]);
   }
  printf("\n");
  }
}

float resolve_seq(int m, int n, int np, float td, float h) {
  double timeStart, timeEnd, Texec;
  struct timeval tp;
  gettimeofday (&tp, NULL);
  timeStart = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
  int i,j,k,l=0;
  float matrix[2][m][n];
  init_matrix(m,n,matrix);
  show_matrix(0,m,n,matrix[0]);
  for(k=1;k<= np;k++) {
    for(i=1;i<m-1;i++) {
      for(j=1;j<n-1;j++) {
        matrix[l][i][j] = (1-4*td/(h*h)) * matrix[1-l][i][j] + (td/(h*h)) * (matrix[1-l][i - 1][j] + matrix[1-l][i + 1][j] + matrix[1-l][i][j - 1] + matrix[1-l][i][j + 1]);
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

// Fonction de lecture du fichier .cl
// Source https://developer.nvidia.com/opencl
char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength)
{
  // locals
  FILE* pFileStream = NULL;
  size_t szSourceLength;
  // open the OpenCL source code file
  pFileStream = fopen(cFilename, "rb");
  size_t szPreambleLength = strlen(cPreamble);
  // get the length of the source code
  fseek(pFileStream, 0, SEEK_END);
  szSourceLength = ftell(pFileStream);
  fseek(pFileStream, 0, SEEK_SET);
  // allocate a buffer for the source code string and read it in
  char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1);
  memcpy(cSourceString, cPreamble, szPreambleLength);
  if (fread((cSourceString)+szPreambleLength, szSourceLength, 1, pFileStream) != 1)
  {
    fclose(pFileStream);
    free(cSourceString);
    return 0;
  }
  // close the file and return the total length of the combined (preamble + source) string
  fclose(pFileStream);
  if (szFinalLength != 0)
  {
    *szFinalLength = szSourceLength + szPreambleLength;
  }
  cSourceString[szSourceLength + szPreambleLength] = '\0';
  return cSourceString;
}

float resolve_par(int m, int n, int np, float td, float h) {
  double timeStart, timeEnd, Texec;
  struct timeval tp;
  gettimeofday(&tp, NULL);
  timeStart = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
  
  int err,k,l=0;
  char *kernel_src;
  size_t kernel_len;
  float matrix[2][m][n];
  int mat_size = sizeof(float)*m*n;
  float* mat0 = (float*) malloc(mat_size);
  float* mat1 = (float*) malloc(mat_size);
  size_t global_item_size = (m-2)*(n-2);
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem d_mat0, d_mat1;

  // Initialise les 2 etats de la matrice
  init_matrix(m,n,matrix);
  show_matrix(0,m,n,matrix[0]);
  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
      mat0[i*n+j] = matrix[0][i][j];
      mat1[i*n+j] = matrix[1][i][j];
    }
  }

  err = clGetPlatformIDs(1, &platform_id, NULL);
  if (err < 0) {
    perror("Error: Couldn't find any platforms");
    exit(1);
  }

  err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
  if (err < 0) {
    perror("Error: Couldn't find any devices");
    exit(1);
  }

  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  if (err < 0)
  {
    printf("Error: Failed to create a compute context!\n");
    return EXIT_FAILURE;
  }

  command_queue = clCreateCommandQueue(context, device_id, 0, &err);
  if (!command_queue)
  {
    printf("Error: Failed to create a command commands!\n");
    return EXIT_FAILURE;
  }

  kernel_src = oclLoadProgSource("TP4.cl", "", &kernel_len);
  program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src, (const size_t *)&kernel_len, &err);
  if (!program)
  {
    printf("Error: Failed to create compute program!\n");
    return EXIT_FAILURE;
  }

  err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    size_t len;
    char buffer[2048];
    printf("Error: Failed to build program executable!\n");
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer),
    buffer, &len);

    printf("%s\n", buffer);
    exit(1);
  }

  // Copie des matrices vers le device
  d_mat0 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mat_size, mat0, &err);
  d_mat1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mat_size, mat1, &err);
  if (!d_mat0 || !d_mat1)
  {
    printf("Error: Failed to allocate device memory!\n");
    exit(1);
  }

  kernel = clCreateKernel(program, "heat_transfer", &err);
  if (!kernel || err != CL_SUCCESS)
  {
    printf("Error: Failed to create compute kernel!\n");
    exit(1);
  }

  // Arguments du kernel
  err =  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_mat0);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_mat1);
  err |= clSetKernelArg(kernel, 2, sizeof(int), (void *)&m);
  err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&n);
  err |= clSetKernelArg(kernel, 4, sizeof(float), (void *)&td);
  err |= clSetKernelArg(kernel, 5, sizeof(float), (void *)&h);
  if (err != CL_SUCCESS){
    printf("Error: Failed to set kernel arguments! %d\n", err);
    exit(1);
  }


  // Execution du kernel
  for(k=1;k<=np;k++) {
    err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
    if (err)
    {
      printf("Error: Failed to execute kernel!\n");
      return EXIT_FAILURE;
    }
    l = k%2;
    err = clSetKernelArg(kernel, l, sizeof(cl_mem), (void *)&d_mat0);
    err |= clSetKernelArg(kernel, 1-l, sizeof(cl_mem), (void *)&d_mat1);
    if (err != CL_SUCCESS)
    {
      printf("Error: Failed to set kernel arguments! %d\n", err);
      exit(1);
    }
  }

  // Recuperation de la bonne matrice resultat selon le nombre de passes
  err = clEnqueueReadBuffer(command_queue, ((1-np%2) == 0 ? d_mat0 : d_mat1 ), CL_TRUE, 0, mat_size, &matrix[l], 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    exit(1);
  }
  show_matrix(1,m,n,matrix[l]);

  err = clFlush(command_queue);
  err = clFinish(command_queue);
  err = clReleaseKernel(kernel);
  err = clReleaseProgram(program);
  err = clReleaseCommandQueue(command_queue);
  err = clReleaseContext(context);
  err = clReleaseMemObject(d_mat0);
  err = clReleaseMemObject(d_mat1);
  free(mat0);
  free(mat1);

  gettimeofday (&tp, NULL);
  timeEnd = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
  Texec = timeEnd - timeStart;
  return Texec;
}

int main (int argc, char *argv[]){
  if (argc == 6){
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int np = atoi(argv[3]);
    float td = atof(argv[4]);
    float h = atof(argv[5]);
    printf("------------------------------\n");
    printf("Version sequentielle\n");
    printf("------------------------------\n\n");
    float texec_seq = resolve_seq(m, n, np, td, h);
    printf("Temps d'execution sequentiel: %f\n", texec_seq);  
    printf("\n------------------------------\n");
    printf("Version parallele\n");
    printf("------------------------------\n\n");
    float texec_par = resolve_par(m, n, np, td, h);
    printf("Temps d'execution parallele: %f\n", texec_par);
    printf("\n------------------------------\n");
    printf("Acceleration: %f\n", texec_seq/texec_par);
    printf("------------------------------\n\n");
  }
  else {
    printf ("Syntaxe: ./TP4 n m np td h\n");
    exit(1);
  }
  return 0;
}
