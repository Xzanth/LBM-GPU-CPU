#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "immintrin.h"
#include "mpi.h"

#define NSPEEDS		9
#define FINALSTATEFILE	"final_state.dat"
#define AVVELSFILE	"av_vels.dat"
#define MASTER		0

#define LOAD(a)		_mm256_load_ps(a)
#define LOADU(a)	_mm256_loadu_ps(a)
#define STORE(a, b)	_mm256_store_ps(a, b)
#define MSTORE(a, b, c)	_mm256_maskstore_ps(a, b, c)

#define SET1(a)		_mm256_set1_ps(a)
#define ADD(a, b)	_mm256_add_ps(a, b)
#define SUB(a, b)	_mm256_sub_ps(a, b)
#define DIV(a, b)	_mm256_div_ps(a, b)
#define MUL(a, b)	_mm256_mul_ps(a, b)
#define SQRT(a)		_mm256_sqrt_ps(a)
#define FMA(a, b, c)	_mm256_fmadd_ps(a, b, c)
#define FNMA(a, b, c)	_mm256_fnmadd_ps(a, b, c)

#define LOADI(a)	_mm256_load_si256(a)
#define STOREI(a, b)	_mm256_store_si256(a, b)
#define CVTIF(a)	_mm256_cvtepi32_ps(a)

#define SUBI(a, b)	_mm256_sub_epi32(a, b)
#define SETI1(a)	_mm256_set1_epi32(a)

float cvtpsf32(__m256 v) {
    __m128 up_four = _mm256_extractf128_ps(v, 1);
    __m128 down_four = _mm256_castps256_ps128(v);
    __m128 four = _mm_add_ps(down_four, up_four);
    __m128 up_two = _mm_movehl_ps(four, four);
    __m128 down_two = four;
    __m128 two = _mm_add_ps(down_two, up_two);
    __m128 up = _mm_shuffle_ps(two, two, 0x1);
    __m128 down = two;
    __m128 single = _mm_add_ss(down, up);
    return _mm_cvtss_f32(single);
}
	
/* struct to hold the parameter values */
typedef struct
{
	int	nx;		/* no. of cells in x-direction */
	int	ny;		/* no. of cells in y-direction */
	int	px;
	int	n;		/* no. of cells in total */
	int	maxIters;	/* no. of iterations */
	int	reynolds_dim;	/* dimension for Reynolds number */
	int	tot_fluid;	/* no. of non blocked cells */
	float	density;	/* density per link */
	float	accel;		/* density redistribution */
	float	omega;		/* relaxation parameter */
} t_param;

typedef struct
{
	int X;
	int Y;
	int T;
} t_dimen;

t_dimen splitgrid(int size)
{
	t_dimen r;

	r.T = size;

	switch (size) {
		case 4:
			r.X = 2;
			r.Y = 2;
			break;
		case 14:
			r.X = 2;
			r.Y = 7;
			break;
		case 28:
			r.X = 4;
			r.Y = 7;
			break;
		case 42:
			r.X = 7;
			r.Y = 6;
			break;
		case 56:
			r.X = 8;
			r.Y = 7;
			break;
		case 70:
			r.X = 7;
			r.Y = 10;
			break;
		case 84: 
			r.X = 7;
			r.Y = 12;
			break;
		case 98:
			r.X = 7;
			r.Y = 14;
			break;
		case 112:
			r.X = 8;
			r.Y = 14;
			break;
		default:
			r.X = 1;
			r.Y = size;
	}

	return r;
}

typedef struct {
	int startx;
	int endx;
	int starty;
	int endy;

} t_block;

			
/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile, t_param* params,
		float** cells_ptr, float** tmp_cells_ptr, float** obstacles_ptr,
		float** av_vels_ptr, float** buff1_ptr, float** buff2_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, float* cells, float* tmp_cells, float* obstacles, float* buffer1, float* buffer2, t_block* blocks, int rank, t_dimen size);
float collision(const t_param params, float* cells, float* tmp_cells, float* obstacles, float* buffer1, float* buffer2, t_block* blocks, int rank, t_dimen size);
int accelerate_flow(const t_param params, float* cells, float* obstacles);
int write_values(const t_param params, float* cells, float* obstacles, float* av_vels);

int combine(const t_param params, float* cells, t_block* blocks, int rank, t_dimen size);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
		 float** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, float* cells);

/* compute average velocity */
float av_velocity(const t_param params, float* cells, float* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, float* cells, float* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

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
	float*	buffer1		= NULL;
	float*	buffer2		= NULL;
	t_param params;			/* struct to hold parameter values */
	struct	timeval timstr;		/* structure to hold elapsed time */
	struct	rusage ru;		/* structure to hold CPU time--system and user */
	double	tic, toc;		/* floating point numbers to calculate elapsed wallclock time */
	double	usrtim;			/* floating point number to record elapsed user CPU time */
	double	systim;			/* floating point number to record elapsed system CPU time */
	int	rank;
	int	size;

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

	/* initialise our data structures and load values from file */
	initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &buffer1, &buffer2);

	t_dimen sizes = splitgrid(size);
	int xsize = params.nx / 8;
	int xquo = xsize / sizes.X; // 9
	int xrem = xsize % sizes.X; // 3
	int xnum = sizes.X - xrem;	// 4

	int ysize = params.ny;
	int yquo = ysize / sizes.Y; // 9
	int yrem = ysize % sizes.Y; // 3
	int ynum = sizes.Y - yrem;	// 4

	t_block blocks[size];

	for (int i = 0; i < sizes.Y; i++) {
		for (int j = 0; j < sizes.X; j++) {
			int xstart = blocks[i * sizes.X + j].startx = 8 * (xquo * j + ((j > xnum) ? j - xnum : 0));
			int ystart = blocks[i * sizes.X + j].starty = yquo * i + ((i > ynum) ? i - ynum : 0);

			blocks[i * sizes.X + j].endx = xstart + 8 * (xquo + ((j >= xnum) ? 1 : 0));
			blocks[i * sizes.X + j].endy = ystart + yquo + ((i >= ynum) ? 1 : 0);
		}
	}

	if (rank == MASTER) {
		gettimeofday(&timstr, NULL);
		tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
	}
	for (int tt = 0; tt < params.maxIters; tt++)
	{
		float* ptrtmp;
		ptrtmp = cells;
		cells = tmp_cells;
		tmp_cells = ptrtmp;
		av_vels[tt] = timestep(params, cells, tmp_cells, obstacles, buffer1, buffer2, blocks, rank, sizes);
#ifdef DEBUG
		printf("==timestep: %d==\n", tt);
		printf("av velocity: %.12E\n", av_vels[tt]);
		printf("tot density: %.12E\n", total_density(params, cells));
#endif
	}
	

	combine(params, cells, blocks, rank, sizes);

	if (rank == MASTER) {
		gettimeofday(&timstr, NULL);
		toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
		getrusage(RUSAGE_SELF, &ru);
		timstr = ru.ru_utime;
		usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
		timstr = ru.ru_stime;
		systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

		printf("==done==\n");
		printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
		printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
		printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
		printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
		write_values(params, cells, obstacles, av_vels);
	}

	finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

	MPI_Finalize();
	return EXIT_SUCCESS;
}

float timestep(const t_param params, float* cells, float* tmp_cells, float* obstacles, float* buffer1, float* buffer2, t_block* blocks, int rank, t_dimen size)
{
	//if ((params.ny - 2) >= start[rank] && (params.ny - 2) < end[rank]) accelerate_flow(params, tmp_cells, obstacles);
	accelerate_flow(params, tmp_cells, obstacles);
	return collision(params, cells, tmp_cells, obstacles, buffer1, buffer2, blocks, rank, size);
}

int accelerate_flow(const t_param params, float* cells, float* obstacles)
{
	/* compute weighting factors */
	float w1 = params.density * params.accel / 9.f;
	float w2 = params.density * params.accel / 36.f;

	/* modify the 2nd row of the grid */
	int jj = params.ny + 6;

	for (int ii = 8; ii < (params.nx + 8); ii++)
	{
		/* if the cell is not occupied and
		** we don't send a negative density */
		if (!obstacles[ii + jj*params.px]
				&& (cells[ii + jj*params.px + (3 * params.n)] - w1) > 0.f
				&& (cells[ii + jj*params.px + (6 * params.n)] - w2) > 0.f
				&& (cells[ii + jj*params.px + (7 * params.n)] - w2) > 0.f)
		{
			/* increase 'east-side' densities */
			cells[ii + jj*params.px + (1 * params.n)] += w1;
			cells[ii + jj*params.px + (5 * params.n)] += w2;
			cells[ii + jj*params.px + (8 * params.n)] += w2;
			/* decrease 'west-side' densities */
			cells[ii + jj*params.px + (3 * params.n)] -= w1;
			cells[ii + jj*params.px + (6 * params.n)] -= w2;
			cells[ii + jj*params.px + (7 * params.n)] -= w2;
		}
	}

	return EXIT_SUCCESS;
}

float collision(const t_param params, float* cells, float* tmp_cells, float* obstacles, float* buffer1, float* buffer2, t_block* blocks, int rank, t_dimen sizes)
{
	const float w0 = 4.f / 9.f;  /* weighting factor */
	const float w1 = 1.f / 9.f;  /* weighting factor */
	const float w2 = 1.f / 36.f; /* weighting factor */

	//float tmp_cell_0, tmp_cell_1, tmp_cell_2, tmp_cell_3, tmp_cell_4, tmp_cell_5, tmp_cell_6, tmp_cell_7, tmp_cell_8;
	float u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8;
	float d_equ_0, d_equ_1, d_equ_2, d_equ_3, d_equ_4, d_equ_5, d_equ_6, d_equ_7, d_equ_8;
	float c_1, c_2;
	float local_d[params.px] __attribute__((aligned(32))), u_x[params.px] __attribute__((aligned(32))), u_y[params.px] __attribute__((aligned(32))), u_sq[params.px] __attribute__((aligned(32)));
	float omega_c = 1.f - params.omega;
	float omega = params.omega;
	
	int tot_cells = 0; /* no. of cells used in calculation */
	__m256 tot_v = SET1(0.f);
	float tot_u = 0.f; /* accumulated magnitudes of velocity for each cell */

	int X = rank % sizes.X;
	int Y = rank / sizes.X;
	int Xlen = blocks[rank].endx - blocks[rank].startx;
	int Ylen = blocks[rank].endy - blocks[rank].starty;

	int top = Y - 1 == -1 ? X + sizes.X * (sizes.Y - 1) : rank - sizes.X;
	int bottom = Y + 1 == sizes.Y ? X : rank + sizes.X;
	int left = X - 1 == -1 ? rank + sizes.X - 1 : rank - 1;
	int right = X + 1 == sizes.X ? rank - sizes.X + 1 : rank + 1;

	MPI_Datatype rowtype, smallcol, coltype;

	MPI_Type_vector(Ylen, 1, params.px, MPI_FLOAT, &smallcol);
	MPI_Type_vector(9, Xlen + 2, params.n, MPI_FLOAT, &rowtype);
	MPI_Type_create_hvector(9, 1, params.n * sizeof(float), smallcol, &coltype);

	MPI_Type_commit(&rowtype);
	MPI_Type_commit(&smallcol);
	MPI_Type_commit(&coltype);

	float* buffer3 = buffer1 + (18 * (Xlen + 2));
	float* buffer4 = buffer2 + (18 * (Xlen + 2));

	if ((sizes.X * sizes.Y) > 1) {
		if (X % 2 == 0) {
			MPI_Send(&tmp_cells[8 + blocks[rank].startx + (8 + blocks[rank].starty)*params.px], 1, coltype, left, 0, MPI_COMM_WORLD);
			MPI_Recv(&tmp_cells[7 + blocks[rank].startx + (8 + blocks[rank].starty)*params.px], 1, coltype, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(&tmp_cells[7 + blocks[rank].endx   + (8 + blocks[rank].starty)*params.px], 1, coltype, right, 0, MPI_COMM_WORLD);
			MPI_Recv(&tmp_cells[8 + blocks[rank].endx   + (8 + blocks[rank].starty)*params.px], 1, coltype, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else {
			MPI_Recv(&tmp_cells[8 + blocks[rank].endx   + (8 + blocks[rank].starty)*params.px], 1, coltype, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(&tmp_cells[7 + blocks[rank].endx   + (8 + blocks[rank].starty)*params.px], 1, coltype, right, 0, MPI_COMM_WORLD);
			MPI_Recv(&tmp_cells[7 + blocks[rank].startx + (8 + blocks[rank].starty)*params.px], 1, coltype, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(&tmp_cells[8 + blocks[rank].startx + (8 + blocks[rank].starty)*params.px], 1, coltype, left, 0, MPI_COMM_WORLD);
		}
		if (Y % 2 == 0) {
			MPI_Send(&tmp_cells[7 + blocks[rank].startx + (8 + blocks[rank].starty)*params.px], 1, rowtype, top, 0, MPI_COMM_WORLD);
			MPI_Recv(&tmp_cells[7 + blocks[rank].startx + (7 + blocks[rank].starty)*params.px], 1, rowtype, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(&tmp_cells[7 + blocks[rank].startx + (7 + blocks[rank].endy)*params.px], 1, rowtype, bottom, 0, MPI_COMM_WORLD);
			MPI_Recv(&tmp_cells[7 + blocks[rank].startx + (8 + blocks[rank].endy)*params.px], 1, rowtype, bottom, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else {
			MPI_Recv(&tmp_cells[7 + blocks[rank].startx + (8 + blocks[rank].endy)*params.px], 1, rowtype, bottom, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(&tmp_cells[7 + blocks[rank].startx + (7 + blocks[rank].endy)*params.px], 1, rowtype, bottom, 0, MPI_COMM_WORLD);
			MPI_Recv(&tmp_cells[7 + blocks[rank].startx + (7 + blocks[rank].starty)*params.px], 1, rowtype, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(&tmp_cells[7 + blocks[rank].startx + (8 + blocks[rank].starty)*params.px], 1, rowtype, top, 0, MPI_COMM_WORLD);
		}
	}

	for (int j = 8 + blocks[rank].starty; j < (blocks[rank].endy + 8); j++) {
		for (int i = 8 + blocks[rank].startx; i < (blocks[rank].endx + 8); i += 8 ) {

			int y_n = j + 1;
			int x_e = i + 1;
			int y_s = j - 1;
			int x_w = i - 1;

			__m256 tmp_cell_0 = LOAD (&tmp_cells[i   +   j*params.px + (0 * params.n)]);
			__m256 tmp_cell_1 = LOADU(&tmp_cells[x_w +   j*params.px + (1 * params.n)]);
			__m256 tmp_cell_2 = LOAD (&tmp_cells[i   + y_s*params.px + (2 * params.n)]);
			__m256 tmp_cell_3 = LOADU(&tmp_cells[x_e +   j*params.px + (3 * params.n)]);
			__m256 tmp_cell_4 = LOAD (&tmp_cells[i   + y_n*params.px + (4 * params.n)]);
			__m256 tmp_cell_5 = LOADU(&tmp_cells[x_w + y_s*params.px + (5 * params.n)]);
			__m256 tmp_cell_6 = LOADU(&tmp_cells[x_e + y_s*params.px + (6 * params.n)]);
			__m256 tmp_cell_7 = LOADU(&tmp_cells[x_e + y_n*params.px + (7 * params.n)]);
			__m256 tmp_cell_8 = LOADU(&tmp_cells[x_w + y_n*params.px + (8 * params.n)]);

			__m256 v1_5_8 = ADD(ADD(tmp_cell_1, tmp_cell_5), tmp_cell_8);
			__m256 v3_6_7 = ADD(ADD(tmp_cell_3, tmp_cell_6), tmp_cell_7);
			__m256 v2_5_6 = ADD(ADD(tmp_cell_2, tmp_cell_5), tmp_cell_6);
			__m256 v4_7_8 = ADD(ADD(tmp_cell_4, tmp_cell_7), tmp_cell_8);

			__m256 loc_d = ADD(
					ADD(ADD(tmp_cell_0, tmp_cell_2), tmp_cell_4),
					ADD(v1_5_8, v3_6_7));

			STORE(&local_d[i], loc_d);


			__m256 ux = DIV(SUB(v1_5_8, v3_6_7), loc_d);
			__m256 uy = DIV(SUB(v2_5_6, v4_7_8), loc_d);

			STORE(&u_x[i], ux);
			STORE(&u_y[i], uy);
			STORE(&u_sq[i], FMA(uy, uy, MUL(ux, ux)));
		}

		for (int i = 8 + blocks[rank].startx; i < (blocks[rank].endx + 8); i += 8) {

			int y_n = j + 1;
			int x_e = i + 1;
			int y_s = j - 1;
			int x_w = i - 1;

			__m256 b_x = LOAD(&u_x[i]);
			__m256 b_y = LOAD(&u_y[i]);

			__m256 tmp_cell_0 = LOAD (&tmp_cells[i   +   j*params.px + (0 * params.n)]);
			__m256 tmp_cell_1 = LOADU(&tmp_cells[x_w +   j*params.px + (1 * params.n)]);
			__m256 tmp_cell_2 = LOAD (&tmp_cells[i   + y_s*params.px + (2 * params.n)]);
			__m256 tmp_cell_3 = LOADU(&tmp_cells[x_e +   j*params.px + (3 * params.n)]);
			__m256 tmp_cell_4 = LOAD (&tmp_cells[i   + y_n*params.px + (4 * params.n)]);

			__m256 u1 = b_x;
			__m256 u2 = b_y;
			__m256 u3 = SUB(SET1(0.f), b_x);
			__m256 u4 = SUB(SET1(0.f), b_y);

			__m256 ld = LOAD(&local_d[i]);
			__m256 usq = LOAD(&u_sq[i]);

			__m256 c1 = MUL(ld, SET1(w1));
			__m256 c2 = FNMA(SET1(1.5f), usq, SET1(1.f));

			__m256 de0 = MUL(MUL(SET1(w0), ld), c2);

			__m256 de1 = MUL(c1, FMA(FMA(u1, SET1(4.5f), SET1(3.f)), u1, c2));
			__m256 de2 = MUL(c1, FMA(FMA(u2, SET1(4.5f), SET1(3.f)), u2, c2));
			__m256 de3 = MUL(c1, FMA(FMA(u3, SET1(4.5f), SET1(3.f)), u3, c2));
			__m256 de4 = MUL(c1, FMA(FMA(u4, SET1(4.5f), SET1(3.f)), u4, c2));

			__m256 cell0 = FMA(tmp_cell_0, SET1(omega_c), MUL(SET1(omega), de0));
			__m256 cell1 = FMA(tmp_cell_1, SET1(omega_c), MUL(SET1(omega), de1));
			__m256 cell2 = FMA(tmp_cell_2, SET1(omega_c), MUL(SET1(omega), de2));
			__m256 cell3 = FMA(tmp_cell_3, SET1(omega_c), MUL(SET1(omega), de3));
			__m256 cell4 = FMA(tmp_cell_4, SET1(omega_c), MUL(SET1(omega), de4));
			
			__m256 obst = LOAD(&obstacles[i + j*params.px]);

			STORE(&cells[i + j*params.px + (0 * params.n)], FMA(tmp_cell_0, obst, MUL(cell0, SUB(SET1(1.f), obst))));
			STORE(&cells[i + j*params.px + (1 * params.n)], FMA(tmp_cell_3, obst, MUL(cell1, SUB(SET1(1.f), obst))));
			STORE(&cells[i + j*params.px + (2 * params.n)], FMA(tmp_cell_4, obst, MUL(cell2, SUB(SET1(1.f), obst))));
			STORE(&cells[i + j*params.px + (3 * params.n)], FMA(tmp_cell_1, obst, MUL(cell3, SUB(SET1(1.f), obst))));
			STORE(&cells[i + j*params.px + (4 * params.n)], FMA(tmp_cell_2, obst, MUL(cell4, SUB(SET1(1.f), obst))));
		}
		for (int i = 8 + blocks[rank].startx; i < (blocks[rank].endx + 8); i += 8) {
			int y_n = j + 1;
			int x_e = i + 1;
			int y_s = j - 1;
			int x_w = i - 1;

			__m256 b_x = LOAD(&u_x[i]);
			__m256 b_y = LOAD(&u_y[i]);

			__m256 tmp_cell_5 = LOADU(&tmp_cells[x_w + y_s*params.px + (5 * params.n)]);
			__m256 tmp_cell_6 = LOADU(&tmp_cells[x_e + y_s*params.px + (6 * params.n)]);
			__m256 tmp_cell_7 = LOADU(&tmp_cells[x_e + y_n*params.px + (7 * params.n)]);
			__m256 tmp_cell_8 = LOADU(&tmp_cells[x_w + y_n*params.px + (8 * params.n)]);

			__m256 u5 = ADD(b_x, b_y);
			__m256 u6 = SUB(b_y, b_x);
			__m256 u7 = SUB(SET1(0.f), u5);
			__m256 u8 = SUB(b_x, b_y);

			__m256 ld = LOAD(&local_d[i]);
			__m256 usq = LOAD(&u_sq[i]);

			__m256 c1 = MUL(ld, SET1(w2));
			__m256 c2 = FNMA(SET1(1.5f), usq, SET1(1.f));

			__m256 de5 = MUL(c1, FMA(FMA(u5, SET1(4.5f), SET1(3.f)), u5, c2));
			__m256 de6 = MUL(c1, FMA(FMA(u6, SET1(4.5f), SET1(3.f)), u6, c2));
			__m256 de7 = MUL(c1, FMA(FMA(u7, SET1(4.5f), SET1(3.f)), u7, c2));
			__m256 de8 = MUL(c1, FMA(FMA(u8, SET1(4.5f), SET1(3.f)), u8, c2));

			__m256 cell5 = FMA(tmp_cell_5, SET1(omega_c), MUL(SET1(omega), de5));
			__m256 cell6 = FMA(tmp_cell_6, SET1(omega_c), MUL(SET1(omega), de6));
			__m256 cell7 = FMA(tmp_cell_7, SET1(omega_c), MUL(SET1(omega), de7));
			__m256 cell8 = FMA(tmp_cell_8, SET1(omega_c), MUL(SET1(omega), de8));
			
			__m256 obst = LOAD(&obstacles[i + j*params.px]);

			STORE(&cells[i + j*params.px + (5 * params.n)], FMA(tmp_cell_7, obst, MUL(cell5, SUB(SET1(1.f), obst))));
			STORE(&cells[i + j*params.px + (6 * params.n)], FMA(tmp_cell_8, obst, MUL(cell6, SUB(SET1(1.f), obst))));
			STORE(&cells[i + j*params.px + (7 * params.n)], FMA(tmp_cell_5, obst, MUL(cell7, SUB(SET1(1.f), obst))));
			STORE(&cells[i + j*params.px + (8 * params.n)], FMA(tmp_cell_6, obst, MUL(cell8, SUB(SET1(1.f), obst))));
		}
		for (int i = 8 + blocks[rank].startx; i < (blocks[rank].endx + 8); i += 8) {
			__m256 obst = SUB(SET1(1.f), LOAD(&obstacles[i + j*params.px]));

			__m256 cell_0 = LOAD(&cells[i + j*params.px + (0 * params.n)]);
			__m256 cell_1 = LOAD(&cells[i + j*params.px + (1 * params.n)]);
			__m256 cell_2 = LOAD(&cells[i + j*params.px + (2 * params.n)]);
			__m256 cell_3 = LOAD(&cells[i + j*params.px + (3 * params.n)]);
			__m256 cell_4 = LOAD(&cells[i + j*params.px + (4 * params.n)]);
			__m256 cell_5 = LOAD(&cells[i + j*params.px + (5 * params.n)]);
			__m256 cell_6 = LOAD(&cells[i + j*params.px + (6 * params.n)]);
			__m256 cell_7 = LOAD(&cells[i + j*params.px + (7 * params.n)]);
			__m256 cell_8 = LOAD(&cells[i + j*params.px + (8 * params.n)]);

			__m256 v1_5_8 = ADD(ADD(cell_1, cell_5), cell_8);
			__m256 v3_6_7 = ADD(ADD(cell_3, cell_6), cell_7);
			__m256 v2_5_6 = ADD(ADD(cell_2, cell_5), cell_6);
			__m256 v4_7_8 = ADD(ADD(cell_4, cell_7), cell_8);

			__m256 loc_d = ADD(
					ADD(ADD(cell_0, cell_2), cell_4),
					ADD(v1_5_8, v3_6_7));

			__m256 ux = DIV(SUB(v1_5_8, v3_6_7), loc_d);
			__m256 uy = DIV(SUB(v2_5_6, v4_7_8), loc_d);

			tot_v = ADD(tot_v, SQRT(MUL(FMA(ux, ux, MUL(uy, uy)), obst)));
		}

	}

	float temp_u = cvtpsf32(tot_v);
	
	MPI_Reduce(&temp_u, &tot_u, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Type_free(&rowtype);
	MPI_Type_free(&coltype);
	MPI_Type_free(&smallcol);

	return tot_u / (float)params.tot_fluid;
}

int combine(const t_param params, float* cells, t_block* blocks, int rank, t_dimen sizes)
{
	MPI_Datatype smallsq, sqtype;

	for (int i = 1; i < sizes.T; i++) {
		int Ylen = blocks[i].endy - blocks[i].starty;
		int Xlen = blocks[i].endx - blocks[i].startx;

		MPI_Type_vector(Ylen, Xlen, params.px, MPI_FLOAT, &smallsq);
		MPI_Type_create_hvector(9, 1, sizeof(float) * params.n, smallsq, &sqtype);

		MPI_Type_commit(&smallsq);
		MPI_Type_commit(&sqtype);

		if (rank == MASTER) {
			MPI_Recv(&cells[8 + blocks[i].startx + (8 + blocks[i].starty)*params.px], 1, sqtype, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else if (rank == i) {
			MPI_Send(&cells[8 + blocks[i].startx + (8 + blocks[i].starty)*params.px], 1, sqtype, MASTER, 0, MPI_COMM_WORLD);
		}

		MPI_Type_free(&smallsq);
		MPI_Type_free(&sqtype);
	}

	return EXIT_SUCCESS;
}

float av_velocity(const t_param params, float* cells, float* obstacles)
{
	int		 tot_cells = 0;  /* no. of cells used in calculation */
	float tot_u;					/* accumulated magnitudes of velocity for each cell */

	/* initialise */
	tot_u = 0.f;

	/* loop over all non-blocked cells */
	for (int jj = 8; jj < (params.ny + 8); jj++)
	{
		for (int ii = 8; ii < (params.nx + 8); ii++)
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
				++tot_cells;
			}
		}
	}

	return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile, t_param* params,
		float** cells_ptr, float** tmp_cells_ptr, float** obstacles_ptr,
		float** av_vels_ptr, float** buff1_ptr, float** buff2_ptr)
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

	params->px = params->nx + 16;
	params->n = params->px * (params->ny + 16);

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

	/* main grid */
	*cells_ptr = (float*)_mm_malloc(sizeof(float) * 9 * ((16 + params->ny) * (16 + params->nx)), 32);
	*buff1_ptr = (float*)_mm_malloc(sizeof(float) * 2 * 9 * (2 * (16 + params->nx)), 32);
	*buff2_ptr = (float*)_mm_malloc(sizeof(float) * 2 * 9 * (2 * (16 + params->nx)), 32);

	if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

	/* 'helper' grid, used as scratch space */
	*tmp_cells_ptr = (float*)_mm_malloc(sizeof(float) * 9 * ((16 + params->ny) * (16 + params->nx)), 32);

	if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

	/* the map of obstacles */
	*obstacles_ptr = _mm_malloc(sizeof(float) * ((16 + params->ny) * (16 + params->nx)), 32);

	if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

	/* initialise densities */
	float w0 = params->density * 4.f / 9.f;
	float w1 = params->density / 9.f;
	float w2 = params->density / 36.f;

	for (int jj = 0; jj < (params->ny + 16); jj++) {
		for (int ii = 0; ii < (params->nx + 16); ii++) {
			/* centre */
			(*cells_ptr)[ii + jj*params->px] = w0;
			/* axis directions */
			(*cells_ptr)[ii + jj*params->px + (1 * params->n)] = w1;
			(*cells_ptr)[ii + jj*params->px + (2 * params->n)] = w1;
			(*cells_ptr)[ii + jj*params->px + (3 * params->n)] = w1;
			(*cells_ptr)[ii + jj*params->px + (4 * params->n)] = w1;
			/* diagonals */
			(*cells_ptr)[ii + jj*params->px + (5 * params->n)] = w2;
			(*cells_ptr)[ii + jj*params->px + (6 * params->n)] = w2;
			(*cells_ptr)[ii + jj*params->px + (7 * params->n)] = w2;
			(*cells_ptr)[ii + jj*params->px + (8 * params->n)] = w2;
		}
	}

	/* first set all cells in obstacle array to zero */
	for (int jj = 8; jj < (params->ny + 8); jj++) {
		for (int ii = 8; ii < (params->nx + 8); ii++) {
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
		(*obstacles_ptr)[xx + 8 + (8 + yy)*params->px] = (float)blocked;
		params->tot_fluid--;
	}

	/* and close the file */
	fclose(fp);

	/*
	** allocate space to hold a record of the avarage velocities computed
	** at each timestep
	*/
	*av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

	return EXIT_SUCCESS;
}

int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
		 float** obstacles_ptr, float** av_vels_ptr)
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

	return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, float* cells, float* obstacles)
{
	const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

	return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, float* cells)
{
	float total = 0.f;	/* accumulator */

	for (int jj = 8; jj < (params.ny + 8); jj++)
	{
		for (int ii = 8; ii < (params.nx + 8); ii++)
		{
			for (int kk = 0; kk < NSPEEDS; kk++)
			{
				total += cells[ii + jj*params.px + (kk * params.n)];
			}
		}
	}

	return total;
}

int write_values(const t_param params, float* cells, float* obstacles, float* av_vels)
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

	for (int jj = 8; jj < (params.ny + 8); jj++)
	{
		for (int ii = 8; ii < (params.nx + 8); ii++)
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
			fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", (ii - 8), (jj - 8), u_x, u_y, u, pressure, (int)obstacles[ii * params.px + jj]);
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
