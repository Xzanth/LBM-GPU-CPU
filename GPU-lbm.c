#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "immintrin.h"
#include "mpi.h"
#include <CL/opencl.h>

#define NSPEEDS		9
#define FINALSTATEFILE	"final_state.dat"
#define AVVELSFILE	"av_vels.dat"
#define MASTER		0
#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

/* struct to hold the parameter values */
typedef struct
{
	int	nx;		/* no. of cells in x-direction */
	int	ny;		/* no. of cells in y-direction */
	int	px;
	int	n;		/* no. of cells in total */
	int	pn;
	int	maxIters;	/* no. of iterations */
	int	reynolds_dim;	/* dimension for Reynolds number */
	int	tot_fluid;	/* no. of non blocked cells */
	float	density;	/* density per link */
	float	accel;		/* density redistribution */
	float	omega;		/* relaxation parameter */
} t_param;

/* struct to hold OpenCL objects */
typedef struct
{
  cl_device_id      device;
  cl_context        context;
  cl_command_queue  queue;

  cl_program program;
  cl_kernel  accelerate_flow;
  cl_kernel  collision;

  cl_mem cells;
  cl_mem tmp_cells;
  cl_mem obstacles;
  cl_mem av_vel;

  cl_mem PinnedBufIn;
  cl_mem PinnedBufOut;
  cl_mem DevBufIn;
  cl_mem DevBufOut;

  float* DataIn;
  float* DataOut;

  size_t lsx;
  size_t lsy;
} t_ocl;
			
/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile, t_param* params,
		float** cells_ptr, float** tmp_cells_ptr, float** obstacles_ptr,
		float** av_vels_ptr, t_ocl* ocl, int* start, int* end, int size, int rank);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, float* cells, float* tmp_cells, float* obstacles, int* start, int* end, int rank, t_ocl ocl, int sw, int size, int tt);
int transfer(const t_param params, int* start, int* end, int rank, t_ocl ocl, int size);
int collision(const t_param params, float* cells, float* tmp_cells, float* obstacles, int* start, int* end, int rank, t_ocl ocl, int sw, int tt);
int accelerate_flow(const t_param params, int* start, int* end, int rank, t_ocl ocl, int sw);
int write_values(const t_param params, float* cells, float* obstacles, float* av_vels, int* start, int* end, int size);

int get_velocity(const t_param params, float* av_vels, t_ocl ocl, int* start, int* end, int rank);
int combine(const t_param params, float* cells, float* tmp_cells, int* start, int* end, int rank, int size);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
		 float** obstacles_ptr, float** av_vels_ptr, t_ocl ocl);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, float* cells, int* start, int* end, int size);

/* compute average velocity */
float av_velocity(const t_param params, float* cells, float* obstacles, int* start, int* end, int size);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, float* cells, float* obstacles, int* start, int* end, int size);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

cl_device_id selectOpenCLDevice();

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
	char*	paramfile	= NULL; /* name of the input parameter file */
	char*	obstaclefile	= NULL; /* name of a the input obstacle file */
	float*	cells		= NULL; /* grid containing fluid densities */
	float*	tmp_cells	= NULL; /* scratch space */
	float*	obstacles	= NULL; /* grid indicating which cells are blocked */
	float*	av_vels		= NULL; /* a record of the av. velocity computed for each timestep */
	t_param params;			/* struct to hold parameter values */
	struct	timeval timstr;		/* structure to hold elapsed time */
	struct	rusage ru;		/* structure to hold CPU time--system and user */
	double	tic, toc;		/* floating point numbers to calculate elapsed wallclock time */
	double	usrtim;			/* floating point number to record elapsed user CPU time */
	double	systim;			/* floating point number to record elapsed system CPU time */
	int	rank;
	int	size;
	t_ocl    ocl;                 /* struct to hold OpenCL objects */
	cl_int err;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	/* parse the command line */
	if (argc != 3)
	{
		usage(argv[0]);
	}
	else
	{
		paramfile = argv[1];
		obstaclefile = argv[2];
	}

	int start[size];
	int end[size];
	/* initialise our data structures and load values from file */
	initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &ocl, start, end, size, rank);

	int len = end[rank] - start[rank];
	err = clEnqueueWriteBuffer(
	ocl.queue, ocl.cells, CL_TRUE, 0, sizeof(float) * 9 * (len + 2) * params.px, cells, 0, NULL, NULL);

	err = clEnqueueWriteBuffer(
	ocl.queue, ocl.obstacles, CL_TRUE, 0, sizeof(float) * (len + 2) * params.px, &obstacles[start[rank] * params.px], 0, NULL, NULL);

	int sw = 0;
	if (rank == MASTER) {
		gettimeofday(&timstr, NULL);
		tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
	}
	for (int tt = 0; tt < params.maxIters; tt++)
	{
		sw = tt & 1;
		timestep(params, cells, tmp_cells, obstacles, start, end, rank, ocl, sw, size, tt);
#ifdef DEBUG
		printf("==timestep: %d==\n", tt);
		printf("av velocity: %.12E\n", av_vels[tt]);
		printf("tot density: %.12E\n", total_density(params, cells));
#endif
	}
	
	if (!sw){
		err = clEnqueueReadBuffer(ocl.queue, ocl.tmp_cells, CL_TRUE, 0, sizeof(float) * NSPEEDS * params.pn, cells, 0, NULL, NULL);
	}
	else {
		err = clEnqueueReadBuffer(ocl.queue, ocl.cells, CL_TRUE, 0, sizeof(float) * NSPEEDS * params.pn, cells, 0, NULL, NULL);
	}

	get_velocity(params, av_vels, ocl, start, end, rank);
	combine(params, cells, tmp_cells, start, end, rank, size);

	if (rank == MASTER) {
		gettimeofday(&timstr, NULL);
		toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
		getrusage(RUSAGE_SELF, &ru);
		timstr = ru.ru_utime;
		usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
		timstr = ru.ru_stime;
		systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

		printf("==done==\n");
		printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, tmp_cells, obstacles, start, end, size));
		printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
		printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
		printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
		write_values(params, tmp_cells, obstacles, av_vels, start, end, size);
	}

	finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, ocl);

	MPI_Finalize();
	return EXIT_SUCCESS;
}

int timestep(const t_param params, float* cells, float* tmp_cells, float* obstacles, int* start, int* end, int rank, t_ocl ocl, int sw, int size, int tt)
{
	transfer(params, start, end, rank, ocl, size);
	accelerate_flow(params, start, end, rank, ocl, sw);
	collision(params, cells, tmp_cells, obstacles, start, end, rank, ocl, sw, tt);
	return EXIT_SUCCESS;
}

int transfer(const t_param params, int* start, int* end, int rank, t_ocl ocl, int size)
{
	int top = rank - 1 == -1 ? size - 1 : rank - 1;
	int bottom = (rank + 1) % size;

	if (size > 1) {
		if (rank % 2 == 0) {
			MPI_Send(ocl.DataOut, (9 * params.px), MPI_FLOAT, top, 0, MPI_COMM_WORLD);
			MPI_Recv(ocl.DataIn, (9 * params.px), MPI_FLOAT, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(&ocl.DataOut[9 * params.px], (9 * params.px), MPI_FLOAT, bottom, 0, MPI_COMM_WORLD);
			MPI_Recv(&ocl.DataIn[9 * params.px], (9 * params.px), MPI_FLOAT, bottom, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else {
			MPI_Recv(&ocl.DataIn[9 * params.px], (9 * params.px), MPI_FLOAT, bottom, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(&ocl.DataOut[9 * params.px], (9 * params.px), MPI_FLOAT, bottom, 0, MPI_COMM_WORLD);
			MPI_Recv(ocl.DataIn, (9 * params.px), MPI_FLOAT, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(ocl.DataOut, (9 * params.px), MPI_FLOAT, top, 0, MPI_COMM_WORLD);
		}
	}

	cl_int err = clEnqueueWriteBuffer(ocl.queue, ocl.DevBufIn, CL_TRUE, 0, sizeof(float) * 2 * 9 * params.px, ocl.DataIn, 0, NULL, NULL);
	return EXIT_SUCCESS;
}
int accelerate_flow(const t_param params, int* start, int* end, int rank, t_ocl ocl, int sw)
{
	cl_int err;

	int len = end[rank] - start[rank];
	if (sw) {
		err = clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_mem), &ocl.tmp_cells);
	} else {
		err = clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_mem), &ocl.cells);
	}
	err = clSetKernelArg(ocl.accelerate_flow, 1, sizeof(cl_mem), &ocl.obstacles);
	err = clSetKernelArg(ocl.accelerate_flow, 2, sizeof(cl_int), &len);
	int push = 0;
	if ((params.ny - 1) >= start[rank] && (params.ny - 1) < end[rank]) push = params.ny - 1 - start[rank];
	err = clSetKernelArg(ocl.accelerate_flow, 3, sizeof(cl_int), &push);
	err = clSetKernelArg(ocl.accelerate_flow, 4, sizeof(cl_int), &params.px);
	err = clSetKernelArg(ocl.accelerate_flow, 5, sizeof(cl_int), &params.pn);
	err = clSetKernelArg(ocl.accelerate_flow, 6, sizeof(cl_float), &params.density);
	err = clSetKernelArg(ocl.accelerate_flow, 7, sizeof(cl_float), &params.accel);
	err = clSetKernelArg(ocl.accelerate_flow, 8, sizeof(cl_mem), &ocl.DevBufIn);

	size_t global[1] = {params.nx};
	size_t offset[1] = {0};
	err = clEnqueueNDRangeKernel(ocl.queue, ocl.accelerate_flow, 1, offset, global, NULL, 0, NULL, NULL);

	err = clFinish(ocl.queue);

	return EXIT_SUCCESS;
}

int collision(const t_param params, float* cells, float* tmp_cells, float* obstacles, int* start, int* end, int rank, t_ocl ocl, int sw, int tt)
{
	cl_int err = clSetKernelArg(ocl.collision, sw, sizeof(cl_mem), &ocl.cells);
	err = clSetKernelArg(ocl.collision, !sw, sizeof(cl_mem), &ocl.tmp_cells);
	err = clSetKernelArg(ocl.collision, 2, sizeof(cl_mem), &ocl.obstacles);
	err = clSetKernelArg(ocl.collision, 3, sizeof(cl_int), &params.nx);
	err = clSetKernelArg(ocl.collision, 4, sizeof(cl_int), &params.ny);
	err = clSetKernelArg(ocl.collision, 5, sizeof(cl_int), &params.px);
	err = clSetKernelArg(ocl.collision, 6, sizeof(cl_int), &params.pn);
	err = clSetKernelArg(ocl.collision, 7, sizeof(cl_float), &params.omega);
	err = clSetKernelArg(ocl.collision, 8, sizeof(cl_float) * ocl.lsx * ocl.lsy, NULL);
	err = clSetKernelArg(ocl.collision, 9, sizeof(cl_mem), &ocl.av_vel);
	err = clSetKernelArg(ocl.collision, 10, sizeof(cl_mem), &ocl.DevBufOut);
	err = clSetKernelArg(ocl.collision, 11, sizeof(cl_int), &tt);

	int len = end[rank] - start[rank];
	size_t global[2] = {params.nx, len};
	size_t offset[2] = {0, 1};
	size_t local[2] = {ocl.lsx, ocl.lsy};

	err = clEnqueueNDRangeKernel(ocl.queue, ocl.collision, 2, offset, global, local, 0, NULL, NULL);

	err = clFinish(ocl.queue);

	err = clEnqueueReadBuffer(ocl.queue, ocl.DevBufOut, CL_TRUE, 0, sizeof(float) * 18 * params.px, ocl.DataOut, 0, NULL, NULL);

	return EXIT_SUCCESS;
}

int get_velocity(const t_param params, float* av_vels, t_ocl ocl, int* start, int* end, int rank)
{
	int len = end[rank] - start[rank];
	int lnum = (len / ocl.lsy) * (params.nx / ocl.lsx);
	float temp_vel[lnum * params.maxIters];
	float local_vel[params.maxIters];

	cl_int err;

	err = clEnqueueReadBuffer(ocl.queue, ocl.av_vel, CL_TRUE, 0, sizeof(cl_float)*lnum*params.maxIters, temp_vel, 0, NULL, NULL);

	for (int i = 0; i < params.maxIters; i++) {
		float tot = 0;
		for (int j = 0; j < lnum; j++) {
			tot += temp_vel[i*lnum + j];
		}
		local_vel[i] = tot / (float)params.tot_fluid;
	}

	MPI_Reduce(local_vel, av_vels, params.maxIters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	return EXIT_SUCCESS;
}

int combine(const t_param params, float* cells, float* tmp_cells, int* start, int* end, int rank, int size)
{
	int len = end[rank] - start[rank];
	for (int jj = 1; jj < (len + 1); jj++) {
		for (int ii = 0; ii < (params.nx); ii++) {
			for (int kk = 0; kk < NSPEEDS; kk++) {
				tmp_cells[ii + (jj + start[rank])*params.px + (kk * params.n)] = cells[ii + jj*params.px + (kk * params.pn)];
			}
		}
	}

	for (int i = 1; i < size; i++) {
		int length = end[i] - start[i];
		if (rank == MASTER) {
			for (int k = 0; k < NSPEEDS; k++) {
				MPI_Recv(&tmp_cells[(1 + start[i])*params.px + (k * params.n)], (length * params.px), MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		} else if (rank == i) {
			for (int k = 0; k < NSPEEDS; k++) {
				MPI_Send(&tmp_cells[(1 + start[i])*params.px + (k * params.n)], (length * params.px), MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD);
			}
		}
	}

	return EXIT_SUCCESS;
}

float av_velocity(const t_param params, float* cells, float* obstacles, int* start, int* end, int size)
{
	float tot_u     = 0.f;	

	/* loop over all non-blocked cells */
	for (int jj = 1; jj < (params.ny + 1); jj++)
	{
		for (int ii = 0; ii < (params.nx); ii++)
		{
			/* ignore occupied cells */
			if (!obstacles[ii + jj*params.px])
			{
				/* local density total */
				float local_density = 0.f;

				for (int kk = 0; kk < NSPEEDS; kk++)
				{
					local_density += cells[ii + jj*params.px + (kk * params.n)];
				}

				/* x-component of velocity */
				float u_x = (cells[ii + jj*params.px + (1 * params.n)]
					+ cells[ii + jj*params.px + (5 * params.n)]
					+ cells[ii + jj*params.px + (8 * params.n)]
					- (cells[ii + jj*params.px + (3 * params.n)]
					+ cells[ii + jj*params.px + (6 * params.n)]
					+ cells[ii + jj*params.px + (7 * params.n)]))
					/ local_density;
				/* compute y velocity component */
				float u_y = (cells[ii + jj*params.px + (2 * params.n)]
					+ cells[ii + jj*params.px + (5 * params.n)]
					+ cells[ii + jj*params.px + (6 * params.n)]
					- (cells[ii + jj*params.px + (4 * params.n)]
					+ cells[ii + jj*params.px + (7 * params.n)]
					+ cells[ii + jj*params.px + (8 * params.n)]))
					/ local_density;
				/* accumulate the norm of x- and y- velocity components */
				tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
				/* increase counter of inspected cells */
			}
		}
	}

	return tot_u / (float)params.tot_fluid;
}

int initialise(const char* paramfile, const char* obstaclefile, t_param* params,
		float** cells_ptr, float** tmp_cells_ptr, float** obstacles_ptr,
		float** av_vels_ptr, t_ocl* ocl, int* start, int* end, int size, int rank)
{
	char	message[1024];	/* message buffer */
	FILE*	fp;		/* file pointer */
	int	xx, yy;		/* generic array indices */
	int	blocked;	/* indicates whether a cell is blocked by an obstacle */
	int	retval;		/* to hold return value for checking */

	/* open the parameter file */
	fp = fopen(paramfile, "r");

	if (fp == NULL)
	{
		sprintf(message, "could not open input parameter file: %s", paramfile);
		die(message, __LINE__, __FILE__);
	}

	/* read in the parameter values */
	retval = fscanf(fp, "%d\n", &(params->nx));

	if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

	retval = fscanf(fp, "%d\n", &(params->ny));

	if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

	params->px = params->nx;
	params->n = params->px * (params->ny + 2);

	retval = fscanf(fp, "%d\n", &(params->maxIters));

	if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

	retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

	if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

	retval = fscanf(fp, "%f\n", &(params->density));

	if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

	retval = fscanf(fp, "%f\n", &(params->accel));

	if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

	retval = fscanf(fp, "%f\n", &(params->omega));

	if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

	/* and close up the file */
	fclose(fp);

	/*
	** Allocate memory.
	**
	** Remember C is pass-by-value, so we need to
	** pass pointers into the initialise function.
	**
	** NB we are allocating a 1D array, so that the
	** memory will be contiguous.  We still want to
	** index this memory as if it were a (row major
	** ordered) 2D array, however.	We will perform
	** some arithmetic using the row and column
	** coordinates, inside the square brackets, when
	** we want to access elements of this array.
	**
	** Note also that we are using a structure to
	** hold an array of 'speeds'.  We will allocate
	** a 1D array of these structs.
	*/

	int nsize = params->ny;
	int quo = nsize / size; // 9
	int rem = nsize % size; // 3
	int num = size - rem;	// 4


	for (int i = 0; i < size; i++) {
		start[i] = quo * i + ((i > num) ? i - num : 0);
		end[i] = start[i] + quo + ((i >= num) ? 1 : 0);
	}

	int len = end[rank] - start[rank];

	params->pn = params->px * (len + 2);
	/* main grid */
	*cells_ptr = (float*)_mm_malloc(sizeof(float) * 9 * ((2 + len) * (params->px)), 128);

	if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

	/* 'helper' grid, used as scratch space */
	*tmp_cells_ptr = (float*)_mm_malloc(sizeof(float) * 9 * ((2 + params->ny) * (params->px)), 128);

	if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

	/* the map of obstacles */
	*obstacles_ptr = _mm_malloc(sizeof(float) * ((2 + params->ny) * (params->px)), 128);

	if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

	/* initialise densities */
	float w0 = params->density * 4.f / 9.f;
	float w1 = params->density / 9.f;
	float w2 = params->density / 36.f;

	for (int jj = 0; jj < (len + 2); jj++) {
		for (int ii = 0; ii < (params->px); ii++) {
			/* centre */
			(*cells_ptr)[ii + jj*params->px] = w0;
			/* axis directions */
			(*cells_ptr)[ii + jj*params->px + (1 * params->pn)] = w1;
			(*cells_ptr)[ii + jj*params->px + (2 * params->pn)] = w1;
			(*cells_ptr)[ii + jj*params->px + (3 * params->pn)] = w1;
			(*cells_ptr)[ii + jj*params->px + (4 * params->pn)] = w1;
			/* diagonals */
			(*cells_ptr)[ii + jj*params->px + (5 * params->pn)] = w2;
			(*cells_ptr)[ii + jj*params->px + (6 * params->pn)] = w2;
			(*cells_ptr)[ii + jj*params->px + (7 * params->pn)] = w2;
			(*cells_ptr)[ii + jj*params->px + (8 * params->pn)] = w2;
		}
	}
	/* first set all cells in obstacle array to zero */
	for (int jj = 1; jj < (params->ny + 1); jj++) {
		for (int ii = 0; ii < (params->nx); ii++) {
			(*obstacles_ptr)[ii + jj*params->px] = 0;
		}
	}

	params->tot_fluid = params->ny * params->nx;

	/* open the obstacle data file */
	fp = fopen(obstaclefile, "r");

	if (fp == NULL)
	{
		sprintf(message, "could not open input obstacles file: %s", obstaclefile);
		die(message, __LINE__, __FILE__);
	}

	/* read-in the blocked cells list */
	while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
	{
		/* some checks */
		if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

		if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

		if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

		if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

		/* assign to array */
		(*obstacles_ptr)[xx + (1 + yy)*params->px] = (float)blocked;
		params->tot_fluid--;
	}

	/* and close the file */
	fclose(fp);

	/*
	** allocate space to hold a record of the avarage velocities computed
	** at each timestep
	*/
	*av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

	cl_int err;


	// Load OpenCL kernel source
	fp = fopen("kernels.cl", "r");
	fseek(fp, 0, SEEK_END);
	int srclen = ftell(fp) + 1;

	char *source = (char*)malloc(srclen);
	memset(source, 0, srclen);

	rewind(fp);
	fread(source, 1, srclen, fp);
	fclose(fp);

	cl_uint num_platforms = 0;
	cl_uint total_devices = 0;
	cl_platform_id platforms[8];
	cl_device_id devices[MAX_DEVICES];
	char name[MAX_DEVICE_NAME];

	// Get list of platforms
	err = clGetPlatformIDs(8, platforms, &num_platforms);

	for (cl_uint p = 0; p < num_platforms; p++)
	{
		cl_uint device_index = 0;
		cl_uint num_devices = 0;
		err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, MAX_DEVICES-total_devices, devices+total_devices, &num_devices);
		total_devices += num_devices;
	}

	printf("\nAvailable OpenCL devices:\n");
	for (cl_uint d = 0; d < total_devices; d++)
	{
		clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
		printf("%2d: %s\n", d, name);
	}
	printf("\n");

	cl_uint device_index = 1;

	if (device_index >= total_devices)
	{
		fprintf(stderr, "device index set to %d but only %d devices available\n",
			device_index, total_devices);
		exit(1);
	}

	// Print OpenCL device name
	clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME,
			MAX_DEVICE_NAME, name, NULL);
	printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

	ocl->device = devices[device_index];
	ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
	ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
	ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char**)&source, NULL, &err);
	free(source);

	char* build_flags = "-cl-mad-enable -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math";
	err = clBuildProgram(ocl->program, 1, &ocl->device, build_flags, NULL, NULL);

	if (err == CL_BUILD_PROGRAM_FAILURE)
	{
		size_t sz;
		clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
		char *buildlog = malloc(sz);
		clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, sz, buildlog, NULL);
		fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
		free(buildlog);
	}

	ocl->accelerate_flow = clCreateKernel(ocl->program, "accelerate_flow", &err);
	ocl->collision = clCreateKernel(ocl->program, "collision", &err);

	ocl->lsx = 64;
	ocl->lsy = 8;
	int lnum = (len / ocl->lsy) * (params->nx / ocl->lsx);

	ocl->cells     = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * NSPEEDS * (len + 2) * params->px, NULL, &err);
	ocl->tmp_cells = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * NSPEEDS * (len + 2) * params->px, NULL, &err);
	ocl->obstacles = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,  sizeof(float) * (len + 2) * params->px,           NULL, &err);
	ocl->av_vel    = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY, sizeof(float) * lnum * params->maxIters,          NULL, &err);
	
	size_t bufsz = sizeof(float) * NSPEEDS * 2 * params->px;
	ocl->PinnedBufIn  = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,  bufsz, NULL, &err);
	ocl->PinnedBufOut = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, bufsz, NULL, &err);
	ocl->DevBufIn     = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,                          bufsz, NULL, &err);
	ocl->DevBufOut    = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,                         bufsz, NULL, &err);

	ocl->DataIn  = (float*)clEnqueueMapBuffer(ocl->queue, ocl->PinnedBufIn,  CL_TRUE, CL_MAP_WRITE, 0, bufsz, 0, NULL, NULL, NULL);
	ocl->DataOut = (float*)clEnqueueMapBuffer(ocl->queue, ocl->PinnedBufOut, CL_TRUE, CL_MAP_READ,  0, bufsz, 0, NULL, NULL, NULL);

	for (int ii = 0; ii < (params->px); ii++) {
		ocl->DataOut[ii + (0)*params->px] = w0;
		ocl->DataOut[ii + (0 + 9)*params->px] = w0;

		ocl->DataOut[ii + 1*params->px] = w1;
		ocl->DataOut[ii + 2*params->px] = w1;
		ocl->DataOut[ii + 3*params->px] = w1;
		ocl->DataOut[ii + 4*params->px] = w1;
		ocl->DataOut[ii + (1 + 9)*params->px] = w1;
		ocl->DataOut[ii + (2 + 9)*params->px] = w1;
		ocl->DataOut[ii + (3 + 9)*params->px] = w1;
		ocl->DataOut[ii + (4 + 9)*params->px] = w1;

		ocl->DataOut[ii + 5*params->px] = w2;
		ocl->DataOut[ii + 6*params->px] = w2;
		ocl->DataOut[ii + 7*params->px] = w2;
		ocl->DataOut[ii + 8*params->px] = w2;
		ocl->DataOut[ii + (5 + 9)*params->px] = w2;
		ocl->DataOut[ii + (6 + 9)*params->px] = w2;
		ocl->DataOut[ii + (7 + 9)*params->px] = w2;
		ocl->DataOut[ii + (8 + 9)*params->px] = w2;
	}

	return EXIT_SUCCESS;
}

int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
		 float** obstacles_ptr, float** av_vels_ptr, t_ocl ocl)
{
	/*
	** free up allocated memory
	*/
	_mm_free(*cells_ptr);
	*cells_ptr = NULL;

	_mm_free(*tmp_cells_ptr);
	*tmp_cells_ptr = NULL;

	_mm_free(*obstacles_ptr);
	*obstacles_ptr = NULL;

	free(*av_vels_ptr);
	*av_vels_ptr = NULL;


	clReleaseMemObject(ocl.cells);
	clReleaseMemObject(ocl.tmp_cells);
	clReleaseMemObject(ocl.obstacles);
	clReleaseMemObject(ocl.av_vel);
	clReleaseKernel(ocl.accelerate_flow);
	clReleaseKernel(ocl.collision);

	clReleaseProgram(ocl.program);
	clReleaseCommandQueue(ocl.queue);
	clReleaseContext(ocl.context);

	return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, float* cells, float* obstacles, int* start, int* end, int size)
{
	const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

	return av_velocity(params, cells, obstacles, start, end, size) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, float* cells, int* start, int* end, int size)
{
	float total = 0.f;	/* accumulator */

	/* loop over all non-blocked cells */
	for (int r = 0; r < size; r++)
	{
		int len = end[r] - start[r];
		for (int jj = 1; jj < (len + 1); jj++)
		{
			for (int ii = 0; ii < (params.nx); ii++)
			{
				int n = len * params.px;
				int buf = 0;
				if (r > 0)
					buf = end[r-1] * NSPEEDS * params.px;

				for (int kk = 0; kk < NSPEEDS; kk++)
				{
					total += cells[ii + jj*params.px + (kk * n) + buf];
				}
			}
		}
	}

	return total;
}

int write_values(const t_param params, float* cells, float* obstacles, float* av_vels, int* start, int* end, int size)
{
	FILE* fp;			/* file pointer */
	const float c_sq = 1.f / 3.f;	/* sq. of speed of sound */
	float local_density;		/* per grid cell sum of densities */
	float pressure;			/* fluid pressure in grid cell */
	float u_x;			/* x-component of velocity in grid cell */
	float u_y;			/* y-component of velocity in grid cell */
	float u;			/* norm--root of summed squares--of u_x and u_y */

	fp = fopen(FINALSTATEFILE, "w");

	if (fp == NULL)
	{
		die("could not open file output file", __LINE__, __FILE__);
	}

	for (int jj = 1; jj < (params.ny + 1); jj++)
	{
		for (int ii = 0; ii < (params.nx); ii++)
		{
			/* an occupied cell */
			if (obstacles[ii + jj*params.px])
			{
				u_x = u_y = u = 0.f;
				pressure = params.density * c_sq;
			}
			/* no obstacle */
			else
			{
				local_density = 0.f;

				for (int kk = 0; kk < NSPEEDS; kk++)
				{
					local_density += cells[ii + jj*params.px + (kk * params.n)];
				}

				/* compute x velocity component */
				u_x = (cells[ii + jj*params.px + (1 * params.n)]
					+ cells[ii + jj*params.px + (5 * params.n)]
					+ cells[ii + jj*params.px + (8 * params.n)]
					- (cells[ii + jj*params.px + (3 * params.n)]
					+ cells[ii + jj*params.px + (6 * params.n)]
					+ cells[ii + jj*params.px + (7 * params.n)]))
					/ local_density;
				/* compute y velocity component */
				u_y = (cells[ii + jj*params.px + (2 * params.n)]
					+ cells[ii + jj*params.px + (5 * params.n)]
					+ cells[ii + jj*params.px + (6 * params.n)]
					- (cells[ii + jj*params.px + (4 * params.n)]
					+ cells[ii + jj*params.px + (7 * params.n)]
					+ cells[ii + jj*params.px + (8 * params.n)]))
					/ local_density;
				/* compute norm of velocity */
				u = sqrtf((u_x * u_x) + (u_y * u_y));
				/* compute pressure */
				pressure = local_density * c_sq;
			}

			/* write to file */
			fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj - 1, u_x, u_y, u, pressure, (int)obstacles[ii * params.nx + jj]);
		}
	}

	fclose(fp);

	fp = fopen(AVVELSFILE, "w");

	if (fp == NULL)
	{
		die("could not open file output file", __LINE__, __FILE__);
	}

	for (int ii = 0; ii < params.maxIters; ii++)
	{
		fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
	}

	fclose(fp);

	return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
	fprintf(stderr, "Error at line %d of file %s:\n", line, file);
	fprintf(stderr, "%s\n", message);
	fflush(stderr);
	exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
	fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
	exit(EXIT_FAILURE);
}

