
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "CL/opencl.h"
#include <stdint.h>
#include <ctime>
#include <cstdlib>
#include <cmath>

//===================================================
//FOR ALTERA FPGAs

#include "AOCL_Utils.h"
using namespace aocl_utils;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;

//===================================================
//FOR non-ALTERA Devices
/*
static cl_platform_id * platform = NULL;
static cl_device_id * device = NULL;*/
//===================================================



static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_program program = NULL;

#define MAX_SOURCE_SIZE (0x400000)
#define STRING_BUFFER_LEN 1024
#define INPUT_SIZE 4194304
#define TAPS 128
#define WORK_SIZE 64


// Function prototypes
bool init();
void cleanup();
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name);
static void device_info_uint( cl_device_id device, cl_device_info param, const char* name);
static void device_info_bool( cl_device_id device, cl_device_info param, const char* name);
static void device_info_string( cl_device_id device, cl_device_info param, const char* name);
static void display_device_info( cl_device_id device );
static void error( cl_int status, const char* msg_to_print);

// Entry point.
int main() {
  cl_int status;
  unsigned int wave[1];

  if(!init()) {
    return -1;
  }
  
  //-----------------------------------------------------------------------------------------------------
  // Set the memory elements
  printf("\nAllocating memory on the device.\n");
  
  // Allocate space for vectors	
	float *Input_re = (float*)malloc(sizeof(float)*INPUT_SIZE);
	float *Input_im = (float*)malloc(sizeof(float)*INPUT_SIZE);
	float *Output_re = (float*)malloc(sizeof(float)*INPUT_SIZE);
	float *Output_im = (float*)malloc(sizeof(float)*INPUT_SIZE);
	float *Taps_re = (float*)malloc(sizeof(float)*TAPS);
	float *Taps_im = (float*)malloc(sizeof(float)*TAPS);


	srand ((time(0)));
	
	int magnitude_re, magnitude_im;
	
	//Generate a random input! (also set the initial output as 0) 
	for (int i = 0; i < (INPUT_SIZE); i++){
		
		//between 1 and -1
		Input_re[i] =  ((float(rand())/float(RAND_MAX/2))) - 1 ;
		Input_im[i] =  ((float(rand())/float(RAND_MAX/2))) - 1 ;
		
		//between 3 and -3
		magnitude_re = rand()%6 - 3;
		magnitude_im = rand()%6 - 3;
		
		Input_re[i] = Input_re[i] * pow(10, magnitude_re);
		Input_im[i] = Input_im[i] * pow(10, magnitude_im);

		//printf("re: %e   im:%e\n", Input_re[i], Input_im[i]);
		
		//set the output as 0
		Output_re[i] = 0;
		Output_im[i] = 0;
	}


	for (int i = 0; i < TAPS; i++){
		Taps_re[i]=(float)(i);
		Taps_im[i]=(float)(i+128);
	}

	/*for (int i = 0; i < TAPS; i++){
		if(i<64){
		Taps_re[i]=(float)(1);
		Taps_im[i]=(float)(0);
		}
		else{
		Taps_re[i]=(float)(-1);
		Taps_im[i]=(float)(0);
		}
	}*/
	
	
	// Create memory buffer on the device for the vector
	cl_mem Input_re_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, (INPUT_SIZE) * sizeof(float), NULL, &status);
	error(status, "Failed to create memory for Input_re");
	cl_mem Input_im_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, (INPUT_SIZE) * sizeof(float), NULL, &status);
	error(status, "Failed to create memory for Input_im");

	cl_mem Output_re_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, INPUT_SIZE * sizeof(float), NULL, &status);
	error(status, "Failed to create memory for Output_re");
	cl_mem Output_im_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, INPUT_SIZE * sizeof(float), NULL, &status);
	error(status, "Failed to create memory for Output_im");

	cl_mem Taps_re_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, TAPS * sizeof(float), NULL, &status);
	error(status, "Failed to create memory for Taps_re");
	cl_mem Taps_im_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, TAPS * sizeof(float), NULL, &status);
	error(status, "Failed to create memory for Taps_im");

	cl_mem Wave_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int), NULL, &status);
	error(status, "Failed to create memory for Taps_im");
	
	
	// Copy the buffers ["Input", "Output" and "Taps" (real & imaginary)] to the device
	cl_event event1, event2, event3, event4;

	status = clEnqueueWriteBuffer(queue, Input_re_clmem, CL_TRUE, 0, (INPUT_SIZE) * sizeof(float), Input_re, 0, NULL, &event1);
	error(status, "Failed to copy Input_re");
	status = clEnqueueWriteBuffer(queue, Input_im_clmem, CL_TRUE, 0, (INPUT_SIZE) * sizeof(float), Input_im, 0, NULL, NULL);
	error(status, "Failed to copy Input_im");

	status = clEnqueueWriteBuffer(queue, Output_re_clmem, CL_TRUE, 0, (INPUT_SIZE) * sizeof(float), Output_re, 0, NULL, NULL);
	error(status, "Failed to copy Input_re");
	status = clEnqueueWriteBuffer(queue, Output_im_clmem, CL_TRUE, 0, (INPUT_SIZE) * sizeof(float), Output_im, 0, NULL, NULL);
	error(status, "Failed to copy Input_im");

	status = clEnqueueWriteBuffer(queue, Taps_re_clmem, CL_TRUE, 0, TAPS * sizeof(float), Taps_re, 0, NULL, NULL);
	error(status, "Failed to copy Taps_re");
	status = clEnqueueWriteBuffer(queue, Taps_im_clmem, CL_TRUE, 0, TAPS * sizeof(float), Taps_im, 0, NULL, NULL);
	error(status, "Failed to copy Taps_im");
	
	printf("\nKernel initialization is complete.\n");

	
	//-------------------------------------------------------------------------------------------------------
	//Wave 1
	printf("Launching the kernel (1)...");
	wave[0] = 1;

	status = clEnqueueWriteBuffer(queue, Wave_clmem, CL_TRUE, 0, sizeof(unsigned int), wave, 0, NULL, NULL);
	error(status, "Failed to copy the wave number");

	// Set the arguments of the kernel
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&Input_re_clmem);
	error(status, "Failed to set kernel arg 0");
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&Input_im_clmem);
	error(status, "Failed to set kernel arg 1");
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&Output_re_clmem);
	error(status, "Failed to set kernel arg 2");
	status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&Output_im_clmem);
	error(status, "Failed to set kernel arg 3");
	status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&Taps_re_clmem);
	error(status, "Failed to set kernel arg 4");
	status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&Taps_im_clmem);
	error(status, "Failed to set kernel arg 5");
	status = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&Wave_clmem);
	error(status, "Failed to set kernel arg 6");

	// Configure work set over which the kernel will execute
	size_t wgSize[3] = {WORK_SIZE, 1, 1};
	size_t gSize[3] = {(INPUT_SIZE-TAPS), 1, 1};

	// Launch the kernel
	status = clFinish(queue);
	error(status, "Failed to finish (before execution)");
	status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gSize, wgSize, 0, NULL, &event2);
	error(status, "Failed to launch kernel");
	status = clFinish(queue);
	error(status, "Failed to finish (after execution)");
	
	printf(" Done! \n");

	
	//-------------------------------------------------------------------------------------------------------
	//Wave 2
	printf("Launching the kernel (2)...");
	wave[0] = 2;

	status = clEnqueueWriteBuffer(queue, Wave_clmem, CL_TRUE, 0, sizeof(unsigned int), wave, 0, NULL, NULL);
	error(status, "Failed to copy the wave number");

	// Set the arguments of the kernel
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&Input_re_clmem);
	error(status, "Failed to set kernel arg 0");
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&Input_im_clmem);
	error(status, "Failed to set kernel arg 1");
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&Output_re_clmem);
	error(status, "Failed to set kernel arg 2");
	status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&Output_im_clmem);
	error(status, "Failed to set kernel arg 3");
	status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&Taps_re_clmem);
	error(status, "Failed to set kernel arg 4");
	status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&Taps_im_clmem);
	error(status, "Failed to set kernel arg 5");
	status = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&Wave_clmem);
	error(status, "Failed to set kernel arg 6");

	// Configure work set over which the kernel will execute
	//wgSize[3] = {WORK_SIZE, 1, 1};
	//gSize[3] = {INPUT_SIZE, 1, 1};

	// Launch the kernel
	status = clFinish(queue);
	error(status, "Failed to finish (before execution)");
	status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gSize, wgSize, 0, NULL, &event3);
	error(status, "Failed to launch kernel");
	status = clFinish(queue);
	error(status, "Failed to finish (after execution)");
	
	printf(" Done! \n");
	
	//-------------------------------------------------------------------------------------------------------
	
	// Read the cl memory Output_clmem on device to the host variable Output
	status = clEnqueueReadBuffer(queue, Output_re_clmem, CL_TRUE, 0, INPUT_SIZE * sizeof(float), Output_re, 0, NULL, NULL);
	error(status, "Failed read the variable Output_re");
	status = clEnqueueReadBuffer(queue, Output_im_clmem, CL_TRUE, 0, INPUT_SIZE * sizeof(float), Output_im, 0, NULL, &event4);
	error(status, "Failed read the variable Output_im");

	// Wait for command queue to complete pending events
	status = clFinish(queue);
	error(status, "Failed to finish (3)");
	


	//========================================================================
	//RESULT COMPARISON!
	
	// FIR filter execution
	// y0 = (x0*h63) + (x1*h62) + (x2*h61) + ...                                                  <--- complex numbers!!
	// (x_re + j*x_im) * (h_re + j*h_im) = (x_re*h_re - x_im*h_im) + j*(x_re*h_im + x_im*h_re)    <--- complex multiplication
	
	printf("Comparing with CPU code...");
	float error = 0;
	float max_error = 0;
	float result_aux_re, result_aux_im, error_aux;
	
	for(unsigned int i = 0; i<(INPUT_SIZE-TAPS); i++){
		result_aux_re = 0;
		result_aux_im = 0;
		
		for(unsigned int j = 0; j < TAPS; j++) {
			result_aux_re += ((Input_re[i+j] * Taps_re[TAPS-1-j]) - (Input_im[i+j] * Taps_im[TAPS-1-j]));
			result_aux_im += ((Input_re[i+j] * Taps_im[TAPS-1-j]) + (Input_im[i+j] * Taps_re[TAPS-1-j]));
		}
		
		error_aux = abs((result_aux_re - Output_re[i])/ result_aux_re);
		if(error_aux > max_error) max_error = error_aux;
		error += error_aux;

		error_aux = abs((result_aux_im - Output_im[i])/ result_aux_im);
		if(error_aux > max_error) max_error = error_aux;
		error += error_aux;

		/*if(i<5){
			printf("\n CPU:  %e ** %e     \nOpenCL:   %e ** %e ", result_aux_re, result_aux_im, Output_re[i], Output_im[i]);
		}*/
	}
	
	error = error / ((INPUT_SIZE-TAPS)*2);
	
	printf("Done! \n");
	printf("Average relative error: %e     Maximum relative error: %e\n", error, max_error);
	
	//========================================================================
	
	// Check times
	cl_ulong copy_start, exec_start, exec_end, copy_end, kernel_1_end, kernel_2_start;
	double total_time1, total_time2, total_time3;

	clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START, sizeof(copy_start), &copy_start, NULL);
	clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_START, sizeof(exec_start), &exec_start, NULL);
	clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_END, sizeof(kernel_1_end), &kernel_1_end, NULL);
	clGetEventProfilingInfo(event3, CL_PROFILING_COMMAND_START, sizeof(kernel_2_start), &kernel_2_start, NULL);
	clGetEventProfilingInfo(event3, CL_PROFILING_COMMAND_END, sizeof(exec_end), &exec_end, NULL);
	clGetEventProfilingInfo(event4, CL_PROFILING_COMMAND_END, sizeof(copy_end), &copy_end, NULL);

	total_time1 = copy_end - copy_start;
	total_time2 = exec_end - exec_start;
	total_time3 = kernel_2_start - kernel_1_end;
	printf("\nExecution time in milliseconds (with mem. tx.) = %0.3f ms", (total_time1 / 1000000.0) );
	printf("\nExecution time in milliseconds (without mem. tx.) = %0.3f ms", (total_time2 / 1000000.0) );
	printf("\nMemory transfer time = %0.3f ms\n", ( (total_time1-total_time2) / 1000000.0) );
	printf("\nTime between kernels = %0.3f ms\n", ( (total_time3) / 1000000.0) );


	// Free the resources allocated
	cleanup();
	free(Taps_re);
	free(Taps_im);
	free(Input_re);
	free(Input_im);
	free(Output_re);
	free(Output_im);

  return 0;
}


/////// HELPER FUNCTIONS ///////

bool init() {
  cl_int status;

  //=============================================================================================================
  // For Altera Devices
  if(!setCwdToExeDir()) {
	  return false;
  }

  // Get the OpenCL platform.S
  platform = findPlatform("Altera");
  if(platform == NULL) {
	  printf("ERROR: Unable to find Altera OpenCL platform.\n");
	  return false;
  }
   // User-visible output - Platform information
  {
    char char_buffer[STRING_BUFFER_LEN]; 
    printf("Querying platform for info:\n");
    printf("==========================\n");
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
  }

  // Query the available OpenCL devices.
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;

  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

  // We'll just use the first device.
  device = devices[0];

  // Display some device information.
  display_device_info(device);

  // Create the context.
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  error(status, "Failed to create context");

  // Create the command queue.
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  error(status, "Failed to create command queue");

  // Create the program.   (From ALTERA binary)
  std::string binary_file = getBoardBinaryFile("kernel", device);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  error(status, "Failed to build program");

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  const char *kernel_name = "fir";  // Kernel name, as defined in the CL file #hello_world
  kernel = clCreateKernel(program, kernel_name, &status);
  error(status, "Failed to create kernel");

  return true;
  
  //=============================================================================================================
  //For non-Altera Devices
  /*
  //Set up the Platform       (On a computer with X discrete GPUs: platform[0]=CPU + integrated GPU, platform[x = 1..X]=GPU_x)
  cl_uint num_platforms;
  status = clGetPlatformIDs(0, NULL, &num_platforms);
  error(status, "Failed count the platforms");
  platform = (cl_platform_id *) malloc(sizeof(cl_platform_id)*num_platforms);
  status = clGetPlatformIDs(num_platforms, platform, NULL);
  error(status, "Failed to create the platform list");

  // User-visible output - Platform information
  {
    char char_buffer[STRING_BUFFER_LEN]; 
    printf("Querying platform for info:\n");
    printf("==========================\n");
    clGetPlatformInfo(platform[1], CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
    clGetPlatformInfo(platform[1], CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
    clGetPlatformInfo(platform[1], CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
  }

  // Query the available OpenCL devices.
  cl_uint num_devices;
  status = clGetDeviceIDs( platform[1], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  error(status, "Failed count the devices");
  device = (cl_device_id *) malloc(sizeof(cl_device_id)*num_devices);
  status = clGetDeviceIDs( platform[1], CL_DEVICE_TYPE_GPU, num_devices, device, NULL);
  error(status, "Failed to create the device list");

  // Display some device information.
  display_device_info(device[0]);

  // Create the context.
  context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
  error(status, "Failed to create context");

  // Create the command queue.
  queue = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
  error(status, "Failed to create command queue");

  // Create the program.   (From .cl text file)
  //------------------------------------
  // Kernel file loading:
  FILE *fp;
  const char fileName[] = "./kernel.cl";
  size_t source_size;
  char *source_str;
  errno_t err;

  // Load kernel source file 
  err = fopen_s(&fp, fileName, "r");
  if (err != 0) {
	  fprintf(stderr, "Failed to load kernel.\n");	
	  exit(1);
  }	
  source_str = (char *)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  _fcloseall();
  //------------------------------------
  program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &status);
  error(status, "Failed to create program");

  // Build the program that was just created.
  status = clBuildProgram(program, 1, device, NULL, NULL, NULL);
  error(status, "Failed to build program");

  // Create the kernel 
  kernel = clCreateKernel(program, "fir", &status);
  error(status, "Failed to create kernel");

  return true;*/
}

// Free the resources allocated during initialization
void cleanup() {
  if(kernel) {
    clReleaseKernel(kernel);  
  }
  if(program) {
    clReleaseProgram(program);
  }
  if(queue) {
    clReleaseCommandQueue(queue);
  }
  if(context) {
    clReleaseContext(context);
  }
}

// Helper functions to display parameters returned by OpenCL queries
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name) {
   cl_ulong a;
   clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
   printf("%-40s = %lu\n", name, a);
}
static void device_info_uint( cl_device_id device, cl_device_info param, const char* name) {
   cl_uint a;
   clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
   printf("%-40s = %u\n", name, a);
}
static void device_info_bool( cl_device_id device, cl_device_info param, const char* name) {
   cl_bool a;
   clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
   printf("%-40s = %s\n", name, (a?"true":"false"));
}
static void device_info_string( cl_device_id device, cl_device_info param, const char* name) {
   char a[STRING_BUFFER_LEN]; 
   clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
   printf("%-40s = %s\n", name, a);
}

// Query and display OpenCL information on device and runtime environment
static void display_device_info( cl_device_id device ) {

   printf("Querying device for info:\n");
   printf("========================\n");
   device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
   device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
   device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
   device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
   device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
   device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
   device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
   device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
   device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
   device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
   device_info_ulong(device, CL_DEVICE_LOCAL_MEM_TYPE, "CL_DEVICE_LOCAL_MEM_TYPE");
   device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
   device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
   device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
   device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
   device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
   device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
   device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");

   {
      cl_command_queue_properties ccp;
      clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
      printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)?"true":"false"));
      printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE)?"true":"false"));
   }
}

static void error( cl_int status, const char* msg_to_print){

	if(status != 0){
		printf("ERROR: ");
		printf(msg_to_print);
		printf(" (Error = %d)\n", status);
	}

	return;
}
