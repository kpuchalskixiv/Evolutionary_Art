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

// eval_kernel <<<#mates, 1024, shared_mem>>
__global__ void eval_kernel(
  float* population_images, 
  float* population_losses, //pop_size=blockdim.x
  float* target_img,
  int img_x, int img_y,

  bool children=true,
){

  extern  __shared__  float temp[];

  int mate_img_idx, mate_idx;
  if (children) mate_idx=pop_size + blockIdx.x;
  else          mate_idx=blockIdx.x;
  mate_img_idx=mate_idx*img_x*img_y;

  tid=threadIdx.x;
  pixd=threadIdx.x;
  temp[tid]=0;
  while(pixd<img_x*img_y){
    mate_pixel=population_images[raw_mate_idx*img_x*img_y + pixd];
    org_pixel=target_img[pixd];
    temp[threadIdx.x]+=(mate_pixel-org_pixel)*(mate_pixel-org_pixel);
    pixd+=blockDim.x;    
  }
  __syncthreads(); 

  for (int d=blockDim.x/2; d>0; d=d/2) {
    __syncthreads();  // ensure previous step completed 
    if (tid<d)  temp[tid] += temp[tid+d];
  }
  __syncthreads();  // ensure previous step completed 

  if(tid==0)
    population_losses[blockIdx.x]=temp[0];

}


//draw_kernel<<<#mate, 128>>>
// one block draws one mate
__global__ void draw_kernel(
  float* population,
  float* population_images, 
  int limit, 

  int img_x, int img_y,

  bool children=true, int genotype_length=6)
{
  int mate_idx, raw_mate_idx;
  if (children) raw_mate_idx=pop_size + blockIdx.x;
  else          raw_mate_idx=blockIdx.x;
  mate_idx=raw_mate_idx*mate_size*genotype_length;
  
  int x_ul,y_lu,x_br,y_br;
  float opacity, g;
  for(int r=0, r<limit, r++){ //r for recatangle
    x_ul=(int)(population[mate_idx]*(img_x-1));
    y_ul=(int)(population[mate_idx+1]*(img_y-1));
    x_br=(int)(population[mate_idx+2]*(img_x-1));
    y_br=(int)(population[mate_idx+3]*(img_y-1));
    opacity=c[mate_idx+4];
    g=c[mate_idx+5];

    x_size=x_br-x_ul;
    y_size=y_br-y_ul;

    pixd=threadIdx.x;
    pixd_row=pixd/y_size;
    pixd_col=pixd%x_size;

    rectangle_start=raw_mate_idx*img_x*img_y // idx of mate image
                        +x_ul*img_y + y_ul ; // start of rectangle

    while((pixd_row<y_size) || (pixd_col<x_size)){ // cache misses should occur whilst changing rows
      org_pixel=population_images[  rectangle_start 
                        +pixd_row*img_y + pixd_col  // thread pixel
                        ];
      population_images[rectangle_start
                        +pixd_row*img_y + pixd_col
                        ]=(1-opacity)*org_pixel+opacity*g;
      
      pixd+=blockDim.x;
      pixd_row=pixd/y_size;
      pixd_col=pixd%x_size;
    }
    //ensure previous rectangle has been drawn
    __syncthreads(); 
  }
}