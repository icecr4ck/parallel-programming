__kernel void heat_transfer(__global double* curr_mat, __global double* prev_mat, int m, int n, float td, float h)
{
	int id = get_global_id(0);
	int i = id/(n-2) + 1;
	int j = id%(n-2) + 1;
	int index = i*n+j;
	curr_mat[index] = (1-4*td/(h*h)) * prev_mat[index] + (td/(h*h)) * (prev_mat[index-1] + prev_mat[index+1] + prev_mat[index-n] + prev_mat[index+n]);
}