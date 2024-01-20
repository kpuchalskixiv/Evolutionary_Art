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


// block as parent, thread for each gene (blockDim.x=mate_sizeXgenotype_len threads)
// select parent (blockidx.x), create children
__global__ void selection_kernel(  float* population, float* population_copy, int genotype_length=8)
{
  int parent_id = population_losses[blockIdx.x].index;
  int child_chromosome_id = (pop_size+blockIdx.x)*blockDim.x;
  int parent_chromosome_id=parent_id*blockDim.x;

  int tid=threadIdx.x;
  // copy parent itself
  population_copy[blockIdx.x*mate_size*genotype_length+tid]=population[parent_chromosome_id+tid];
  // create childrens, copies of parent
  for(int i=0; i<children_per_mate; i++){
    population_copy[child_chromosome_id+tid]=population[parent_chromosome_id+tid];
    child_chromosome_id+=blockDim.x;
  }
}



// TODO !!!!
// slowly unlock more circles as iterations progres
__global__ void mutation_kernel(  float* populaion, float* mutation_coefs, int genotype_length=8)
{
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  populaion[tid]*=mutation_coefs[tid];
}

// eval_kernel <<<#mates, 1024, shared_mem>>
__global__ void eval_kernel(
  float* population_images, 
  float* target_img,
  int img_x, int img_y
){

  extern  __shared__  float temp[];
  int mate_img_idx, mate_idx;
  mate_idx=blockIdx.x;
  mate_img_idx=mate_idx*img_x*img_y;

  int tid=threadIdx.x;
  int pixd=threadIdx.x;
  temp[tid]=0;
  __syncthreads(); 
  while(pixd<img_x*img_y){
    temp[tid]+=powf(fabsf(target_img[pixd]-population_images[mate_img_idx + pixd]), 2.0f);    
   // if (population_images[mate_img_idx + pixd]>0)
   //   printf("mate+tid: %d, temp[tid]= %f, mate_pixel: %f\n",mate_idx+tid, temp[tid], population_images[mate_img_idx + pixd]);
    pixd+=blockDim.x;    
      __syncthreads(); 

  }
  __syncthreads(); 
 // printf("mate+tid: %d, temp[tid]= %f\n",mate_idx+tid, temp[tid]);

  for (int d=blockDim.x/2; d>0; d=d/2) {
    __syncthreads();  // ensure previous step completed 
    if (tid<d)  temp[tid] += temp[tid+d];
  }
  __syncthreads();  // ensure previous step completed 

  if(tid==0){
    population_losses[mate_idx].value=temp[0];
    population_losses[mate_idx].index=mate_idx;
  }
  __syncthreads(); 
}

// block per image
__global__ void reset_images_kernel(float* population_images, int x_img, int y_img){
  int img_start=blockIdx.x*x_img*y_img;
  int pixd=threadIdx.x;
  while(pixd<x_img*y_img){
    population_images[img_start+pixd]=0.0f;
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

  int genotype_length=8)
{
  int mate_chromosome_idx, mate_idx;
  mate_idx=blockIdx.x;
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
