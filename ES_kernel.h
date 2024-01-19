// device code
#define pop_size 64  // blockX
#define mate_size 16 // blockX
#define circles_per_thread 1 // blockZ
#define children_per_mate 8
#define parents 8

struct loss_item
{
    float value;
    int index;
};
__device__ __managed__ loss_item population_losses[pop_size+children_per_mate*parents];

// TODO !!!!
// slowly unlock more circles as iterations progres
__global__ void mutate_kernel(long long mate_idx,
	         	      const float* __restrict__ population,
			            float* output_img_idx)
{
  //population as pop_size*circles_per_mate X 7 matrix
  // each circle represented as real number vector of dim=7
}

// eval_kernel <<<#mates, 1024, shared_mem>>
__global__ void eval_kernel(
  float* population_images, 
  float* target_img,
  int img_x, int img_y,

  bool children=false
){

  extern  __shared__  float temp[];
  int mate_img_idx, mate_idx;
  if (children) mate_idx=pop_size + blockIdx.x;
  else          mate_idx=blockIdx.x;
  mate_img_idx=mate_idx*img_x*img_y;

  int tid=threadIdx.x;
  int pixd=threadIdx.x;
  temp[tid]=0;
  int mate_pixel, org_pixel;
  while(pixd<img_x*img_y){
    mate_pixel=population_images[mate_img_idx + pixd];
    org_pixel=target_img[pixd];
    temp[tid]+=(mate_pixel-org_pixel)*(mate_pixel-org_pixel);
    pixd+=blockDim.x;    
  }
  __syncthreads(); 

  for (int d=blockDim.x/2; d>0; d=d/2) {
    __syncthreads();  // ensure previous step completed 
    if (tid<d)  temp[tid] += temp[tid+d];
  }
  __syncthreads();  // ensure previous step completed 

  if(tid==0)
    population_losses[mate_idx].value=temp[0];
    population_losses[mate_idx].index=mate_idx;

}

// block per image
__global__ void reset_images_kernel(float* population_images, int x_img, int y_img){
  int img_start=blockIdx.x*x_img*y_img;
  int pixd=threadIdx.x;
  while(pixd<img_start+x_img*y_img){
    population_images[pixd]=0.0f;
    pixd+=blockDim.x;
  }

}

//draw_kernel<<<#mate, 128>>>
// one block draws one mate
__global__ void draw_kernel(
  float* population,
  float* population_images, 
  int limit, 

  int img_x, int img_y,

  bool children=false, int genotype_length=8)
{
  int mate_chromosome_idx, mate_idx;
  if (children) mate_idx=pop_size + blockIdx.x;
  else          mate_idx=blockIdx.x;
  mate_chromosome_idx=mate_idx*mate_size*genotype_length;
  
  int x_ul,y_ul,x_br,y_br, x_size, y_size, pixd, pixd_row, pixd_col, rectangle_start, org_pixel;
  float opacity, g;
  for(int r=0; r<limit; r++){ //r for recatangle
    x_ul=(int)(population[mate_chromosome_idx]*(img_x-1));
    y_ul=(int)(population[mate_chromosome_idx+1]*(img_y-1));
    x_br=(int)(population[mate_chromosome_idx+2]*(img_x-1));
    y_br=(int)(population[mate_chromosome_idx+3]*(img_y-1));
    opacity=population[mate_chromosome_idx+4];
    g=population[mate_chromosome_idx+5];

    x_ul=min(max(0, x_ul), img_x);
    y_ul=min(max(0, y_ul), img_y);
    x_br=min(max(0, x_br), img_x);
    y_br=min(max(0, y_br), img_y);

    x_size=max(0, x_br-x_ul);
    y_size=max(0, y_br-y_ul);

    pixd=threadIdx.x;
    pixd_row=pixd/y_size;
    pixd_col=pixd%x_size;

    rectangle_start=mate_idx*img_x*img_y // idx of mate image
                        +x_ul*img_y + y_ul ; // start of rectangle

    while((pixd<x_size*y_size)
    && (pixd_row<y_size) 
    && (pixd_col<x_size)){ // cache misses should occur whilst changing rows
      population_images[rectangle_start
                        +pixd_row*img_y + y_ul+pixd_col
                        ]*=(1-opacity);
      population_images[rectangle_start
                        +pixd_row*img_y + y_ul+pixd_col
                        ]+=opacity*g;
      
      pixd+=blockDim.x;
      pixd_row=pixd/y_size;
      pixd_col=pixd%x_size;
     // __syncthreads(); 
    }
    //ensure previous rectangle has been drawn
    __syncthreads(); 
  }
}
