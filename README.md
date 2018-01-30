# OpenCL-FPGA-FIR-Filter
OpenCL code for big FIR Filters, on FPGAs (also works for GPUs, with the commented code).
Minimal performance loss when there are more taps than it fits in a single FPGA pipeline.


Problem type: Compute Bound (several operations per load/store)
                [each filter (complex) tap requires 4 ADD e 4 MUL = 8 FP operations!
                 e.g. for a filter with 64 taps, each output requires 64*8 = 512 = 2^9 FP operations]


                 
                 
////////////////////////////////////////////////////////////////////////////////                 
Simulations: 2^22 floating point elements, 64 taps

BW [GB/s] =  16 [MB] / Exec. Time [ms]

TP [GFLOPS] =  ((2^22 * 2^9 FP ops) / 10^9) /  Exec. Time [s] =  2147,4836 / Exec. Time [ms]

(The exec. time doesn't include the time it requires to move the memory from the host to the device)
                 
/////
@NVIDIA GTX860M:                t = 7.122 ms;  avg bandwidth = 13.48 GB/s;   avg GFLOPS = 301.53

@DE5-Net (Stratix V 5SGXA7):    t = 17.22 ms;  avg bandwidth = 0.09  GB/s;   avg GFLOPS = 124.71
