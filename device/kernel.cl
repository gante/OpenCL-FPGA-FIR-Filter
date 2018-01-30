__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void fir( __global float *x_re, 
						  __global float *x_im, 
						  __global float *Output_re, 
						  __global float *Output_im, 
						  __constant float *h_re, 
						  __constant float *h_im,
						  __constant unsigned int *wave){
//Naive kernel: each thread executes one output value. Every value is loaded from the global memory

// Cuda - blockdim = OpenCL - localsize
// Cuda - griddim*blockdim = OpenCL - globalsize
	
 //Get the indexes of the local item
 /*
 unsigned int tid = get_local_id(0);
 unsigned int wid = get_group_id(0);
 unsigned int dim = get_local_size(0); 
 unsigned int i = wid*dim*2 + tid;
 unsigned int gridSize = get_global_size(0)*2;
 */
 
 unsigned int gid = get_global_id(0);
 unsigned int tid = get_local_id(0);
 unsigned int offset = wave[0];
 

 
 
 //For a workgroup size = 64!
 __local float x_re_local[128]; 
 __local float x_im_local[128];
 
 
 unsigned int read_index = gid + (offset-1)*64;
 x_re_local[tid] = x_re[read_index];
 x_im_local[tid] = x_im[read_index];
 
 //if(tid > 1){
	 
	 read_index = read_index + 64;
	  x_re_local[tid+64] = x_re[read_index];
	  x_im_local[tid+64] = x_im[read_index];
 //}
 
 //To make sure the local memory is properly loaded
 barrier(CLK_LOCAL_MEM_FENCE);
 
 
float y_re = Output_re[gid];
float y_im = Output_im[gid];

//(total number of waves +1) - current wave 
unsigned int tap_init = ((3-offset)*64)-1;
//if(gid == 0) printf ("\n wave = %d, tap_init = %d", wave[0], tap_init);


	// FIR filter execution
	// y0 = (x0*h63) + (x1*h62) + (x2*h61) + ...                                                  <--- complex numbers!!
	// (x_re + j*x_im) * (h_re + j*h_im) = (x_re*h_re - x_im*h_im) + j*(x_re*h_im + x_im*h_re)    <--- complex multiplication

#pragma unroll
for(unsigned int i = 0; i<64; i++){
	
	read_index = tid + i;
	
	y_re = y_re + (x_re_local[read_index]*h_re[tap_init-i] - x_im_local[read_index]*h_im[tap_init-i]);
	y_im = y_im + (x_re_local[read_index]*h_im[tap_init-i] + x_im_local[read_index]*h_re[tap_init-i]);
		
}

Output_re[gid] = y_re;
Output_im[gid] = y_im;

//read_index += wave[0];
};
