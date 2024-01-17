//
// Notes: one thread per node in the 3D block
//

// device code

// TODO !!!!
// slowly unlock more circles as iterations progres
__global__ void mutate_kernel(long long mate_idx,
	         	      const float* __restrict__ population,
			            float* output_img_idx)
{
  //population as pop_size*circles_per_mate X 7 matrix
  // each circle represented as real number vector of dim=7
  circle_idx=threadIdx.x + blockIdx.x*blockDim.x
  circle=population[circle_idx]

}

__global__ void evaluate_kernel(long long mate_idx,
	         	      const float* __restrict__ population,
			            float* output_img_idx)
{
  mate_idx=threadIdx.x + blockIdx.x*blockDim.x
}



