#define NSPEEDS	9

kernel void accelerate_flow(global float* cells, global float* obstacles, int y, int ny, int px, int n, float density, float accel, global float* buffer)
{
	float w1 = density * accel / 9.0;
	float w2 = density * accel / 36.0;

	int jj = ny;

	int ii = get_global_id(0);

	for (int k = 0; k < NSPEEDS; k++)
		cells[ii +              (k * n)] = buffer[ii + (k) * px];

	for (int k = 0; k < NSPEEDS; k++)
		cells[ii + (y + 1)*px + (k * n)] = buffer[ii + (9 + k) * px];

	if (ny > 0 && !obstacles[ii + jj*px]
		&& (cells[ii + jj*px + (3 * n)] - w1) > 0.f
		&& (cells[ii + jj*px + (6 * n)] - w2) > 0.f
		&& (cells[ii + jj*px + (7 * n)] - w2) > 0.f)
	{
		cells[ii + jj*px + (1 * n)] += w1;
		cells[ii + jj*px + (5 * n)] += w2;
		cells[ii + jj*px + (8 * n)] += w2;
		cells[ii + jj*px + (3 * n)] -= w1;
		cells[ii + jj*px + (6 * n)] -= w2;
		cells[ii + jj*px + (7 * n)] -= w2;
	}
}

kernel void collision(global float* cells, global float* tmp_cells, global float* obstacles, int nx, int ny, int px, int n, float omega, local float* tot_u, global float* av_vel, global float* buffer, int tt)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	int y_n = j + 1;
	int x_e = i + 1;
	int y_s = j - 1;
	int x_w = i - 1;

	const float w0 = 4.f / 9.f;  /* weighting factor */
	const float w1 = 1.f / 9.f;  /* weighting factor */
	const float w2 = 1.f / 36.f; /* weighting factor */
	const float omega_c = 1.f - omega;

	float tmp_cell_0 = cells[i   +   j*px + (0 * n)];
	float tmp_cell_1 = cells[x_w +   j*px + (1 * n)];
	float tmp_cell_2 = cells[i   + y_s*px + (2 * n)];
	float tmp_cell_3 = cells[x_e +   j*px + (3 * n)];
	float tmp_cell_4 = cells[i   + y_n*px + (4 * n)];
	float tmp_cell_5 = cells[x_w + y_s*px + (5 * n)];
	float tmp_cell_6 = cells[x_e + y_s*px + (6 * n)];
	float tmp_cell_7 = cells[x_e + y_n*px + (7 * n)];
	float tmp_cell_8 = cells[x_w + y_n*px + (8 * n)];

	float loc_d = tmp_cell_0 + tmp_cell_1 + tmp_cell_2 + tmp_cell_3 + tmp_cell_4 + tmp_cell_5 + tmp_cell_6 + tmp_cell_7 + tmp_cell_8;

	float u_x = (tmp_cell_1 + tmp_cell_5 + tmp_cell_8 - (tmp_cell_3 + tmp_cell_6 + tmp_cell_7)) / loc_d;
	float u_y = (tmp_cell_2 + tmp_cell_5 + tmp_cell_6 - (tmp_cell_4 + tmp_cell_7 + tmp_cell_8)) / loc_d;
	float u_sq = u_y * u_y + u_x * u_x;


	float u0 = 0.f;
	float u1 =   u_x;        /* east */
	float u2 =         u_y;  /* north */
	float u3 = - u_x;        /* west */
	float u4 =       - u_y;  /* south */
	float u5 =   u_x + u_y;  /* north-east */
	float u6 = - u_x + u_y;  /* north-west */
	float u7 = - u_x - u_y;  /* south-west */
	float u8 =   u_x - u_y;  /* south-east */

	float c1 = loc_d * w1;
	float c2 = loc_d * w2;
	float c3 = 1.f - (1.5f * u_sq);

	float de0 = w0 * loc_d * c3;
	float de1 = c1 * (c3 + u1 * (3.f + u1 * 4.5f));
	float de2 = c1 * (c3 + u2 * (3.f + u2 * 4.5f));
	float de3 = c1 * (c3 + u3 * (3.f + u3 * 4.5f));
	float de4 = c1 * (c3 + u4 * (3.f + u4 * 4.5f));
	float de5 = c2 * (c3 + u5 * (3.f + u5 * 4.5f));
	float de6 = c2 * (c3 + u6 * (3.f + u6 * 4.5f));
	float de7 = c2 * (c3 + u7 * (3.f + u7 * 4.5f));
	float de8 = c2 * (c3 + u8 * (3.f + u8 * 4.5f));

	u0 = tmp_cell_0 * omega_c + omega * de0;
	u1 = tmp_cell_1 * omega_c + omega * de1;
	u2 = tmp_cell_2 * omega_c + omega * de2;
	u3 = tmp_cell_3 * omega_c + omega * de3;
	u4 = tmp_cell_4 * omega_c + omega * de4;
	u5 = tmp_cell_5 * omega_c + omega * de5;
	u6 = tmp_cell_6 * omega_c + omega * de6;
	u7 = tmp_cell_7 * omega_c + omega * de7;
	u8 = tmp_cell_8 * omega_c + omega * de8;

	float obst = obstacles[i + j*px];
	float obst_c = 1 - obst;

	tmp_cells[i + j*px + (0 * n)] = tmp_cell_0 * obst + u0 * obst_c;
	tmp_cells[i + j*px + (1 * n)] = tmp_cell_3 * obst + u1 * obst_c;
	tmp_cells[i + j*px + (2 * n)] = tmp_cell_4 * obst + u2 * obst_c;
	tmp_cells[i + j*px + (3 * n)] = tmp_cell_1 * obst + u3 * obst_c;
	tmp_cells[i + j*px + (4 * n)] = tmp_cell_2 * obst + u4 * obst_c;
	tmp_cells[i + j*px + (5 * n)] = tmp_cell_7 * obst + u5 * obst_c;
	tmp_cells[i + j*px + (6 * n)] = tmp_cell_8 * obst + u6 * obst_c;
	tmp_cells[i + j*px + (7 * n)] = tmp_cell_5 * obst + u7 * obst_c;
	tmp_cells[i + j*px + (8 * n)] = tmp_cell_6 * obst + u8 * obst_c;

	if (j == 1)
		for (int k = 0; k < NSPEEDS; k++)
			buffer[i +      k  * px] = tmp_cells[i + j*px + (k * n)];

	if (j == get_global_size(1))
		for (int k = 0; k < NSPEEDS; k++)
			buffer[i + (9 + k) * px] = tmp_cells[i + j*px + (k * n)];

	int lid1 = get_local_id(0);
	int lid2 = get_local_id(1);
	int gid1 = get_group_id(0);
	int gid2 = get_group_id(1);
	int lsz1 = get_local_size(0);
	int lsz2 = get_local_size(1);
	int lszt = lsz1 * lsz2;
	int lid = lid2 * lsz1 + lid1;

    tot_u[lid] = sqrt(u_sq) * obst_c;

	for (uint stride = lszt/2; stride > 0; stride /= 2) {
		barrier(CLK_LOCAL_MEM_FENCE);

		if (lid < stride) {
			tot_u[lid] += tot_u[lid + stride];
		}
	}

	if (lid == 0)
		av_vel[tt * (get_num_groups(0) * get_num_groups(1)) + gid2*get_num_groups(0) + gid1] = tot_u[0];

	barrier(CLK_LOCAL_MEM_FENCE);
}
