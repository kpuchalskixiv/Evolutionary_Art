// device code
#define pop_size 32  // blockX
#define mate_size 204 // blockX
#define children_per_mate 4
#define parents 16
#define genotype_length 5


struct loss_item
{
    float value;
    int index;
};
__device__ __managed__ loss_item population_losses[pop_size+children_per_mate*parents];
__device__ __managed__ int parents_ids[parents];

//<<1,pop_size+children*parents>>
__global__ void fitness_values_kernel(
  float max_mse
){
  extern  __shared__  float temp[];

  int tid=blockDim.x*blockIdx.x+threadIdx.x;
  float fit_val=max_mse-population_losses[tid].value;
  temp[tid]=fit_val;

  __syncthreads();  
  for (int d=blockDim.x/2; d>0; d=d/2) {
    __syncthreads();  // ensure previous step completed 
    if (tid<d)  temp[tid] += temp[tid+d];
  }
  __syncthreads();  // ensure previous step completed 

  population_losses[tid].value=fit_val/temp[0];
}


// <<<pop_size, mate_size&genotype_len>>>
__global__ void population_selection_kernel(  
  float* population, float* population_copy
  )
{
  int mate_id = population_losses[blockIdx.x].index;
  int mate_chromosome_id=mate_id*blockDim.x;

  int tid=threadIdx.x;
  // copy mate to new (ordered) position
  population_copy[blockIdx.x*mate_size*genotype_length+tid]=population[mate_chromosome_id+tid];
}
// block as parent, thread for each gene (blockDim.x=mate_sizeXgenotype_lengththreads)
// select parent (blockidx.x), create children
__global__ void create_children_kernel(  
  float* population, float* population_copy
  )
{
  int parent_id = population_losses[parents_ids[blockIdx.x]].index;
  int child_chromosome_id = (pop_size+blockIdx.x)*blockDim.x;
  int parent_chromosome_id=parent_id*blockDim.x;

  int tid=threadIdx.x;
  // create childrens, copies of parent
  for(int i=0; i<children_per_mate; i++){
    population_copy[child_chromosome_id+tid]=population[parent_chromosome_id+tid];
    child_chromosome_id+=blockDim.x;
  }
}

// block as parent, thread for each gene (blockDim.x=genotype_lengththreads)
__global__ void sigmas_selection_kernel(  
  float* sigmas, float* sigmas_copy
  )
{
  int parent_id = population_losses[parents_ids[blockIdx.x]].index;
  int child_sigma_id = (pop_size+blockIdx.x)*blockDim.x;
  int parent_sigma_id=parent_id*blockDim.x;

  int tid=threadIdx.x;
  // copy parent itself
  sigmas_copy[blockIdx.x*mate_size*genotype_length+tid]=sigmas[parent_sigma_id+tid];
  // create childrens, copies of parent
  for(int i=0; i<children_per_mate; i++){
    sigmas_copy[child_sigma_id+tid]=sigmas[parent_sigma_id+tid];
    child_sigma_id+=blockDim.x;
  }
}

//<<<population, genotype_len>>>
//<<<population, mate_sizeXgenotype_len>>>
__global__ void sigmas_mutation_kernel(  
  float* sigmas, 
  float* mutation_coefs, float* mutation_ifs,
  float scale=0.001f, float mut_prob=0.5f)
{
 // mutate only children
  int tid = blockDim.x*(blockIdx.x + pop_size)+ threadIdx.x;
 // else tid = blockDim.x*blockIdx.x + threadIdx.x;

  if(mutation_ifs[tid]<mut_prob){
    sigmas[tid]+=mutation_coefs[tid];
    sigmas[tid]=fminf(0.1f, fmaxf(sigmas[tid], 0.00001f));
  }
}

// <<<population, no_figures*genotype_len>>>
__global__ void mate_mutation_kernel(  float* population, float* mutation_coefs, float* mutation_ifs, float* sigmas,
  float mut_prob=0.5f)
{
  // mutate only children
  int tid = mate_size*(blockIdx.x + pop_size)+ threadIdx.x;
 // else tid = mate_size*blockIdx.x + threadIdx.x;

  if(mutation_ifs[tid]<mut_prob){
    //population[tid]+=sigmas[blockIdx.x*genotype_length+ (threadIdx.x % genotype_length)]*(mutation_coefs[tid]-0.5f);
    population[tid]+=sigmas[tid]*(mutation_coefs[tid]-0.5f);
    // int(lower_bound)=0
    population[tid]=fmaxf(0.002f, fminf(population[tid], 1.0f));
  }
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
__global__ void reset_values_kernel(float* array, float value, int size){
  int arr_start=blockIdx.x*size;
  int pixd=threadIdx.x;
  while(pixd<size){
    array[arr_start+pixd]=value;
    pixd+=blockDim.x;
  }
}
// fix e.g. opacity by <<<pop_size, mate_size>>>(population, 3, value)
__global__ void fix_param_kernel(float* array, int idx, float value){
  int tid=mate_size*blockIdx.x*genotype_length+mate_size*threadIdx.x;
  array[tid+idx]=value;
}

//draw_kernel<<<#mate, 128>>>
// one block draws one mate

__global__ void draw_kernel(
  float* population,
  float* population_images, 
  int limit, 

  int img_x, int img_y,
  float max_radius
  )
{
  int mate_rectangle_idx, mate_idx;
  mate_idx=blockIdx.x;
  mate_rectangle_idx=mate_idx*mate_size*genotype_length;
  
  int x,y,radius, width, height, pixd, pixd_row, pixd_col, rectangle_start;
  float opacity, g;
  __syncthreads(); 
  for(int r=0; r<limit; r++){ //r for recatangle

    x=(int)(population[mate_rectangle_idx]*(img_x));
    y=(int)(population[mate_rectangle_idx+1]*(img_y));

    radius=(int)(population[mate_rectangle_idx+2]*max_radius);
    opacity=population[mate_rectangle_idx+3];
    g=population[mate_rectangle_idx+4];
  //  __syncthreads(); 
    x=min(max(0, x), img_x-1);
    y=min(max(0, y), img_y-1);

    width=min(radius, img_y-y);
    height=min(radius, img_x-x);
    pixd=threadIdx.x;
    pixd_row=pixd/width;
    pixd_col=pixd%width;

  //  printf("%d %d, %d, %f, %f\n", x,y,radius,opacity, g);

    rectangle_start=mate_idx*img_x*img_y // idx of mate image
                        +x*img_y + y ; // start of rectangle
    int id =rectangle_start +pixd_row*img_y +pixd_col;
    __syncthreads(); 
    while((pixd<width*height)
    && (pixd_row<height) 
    //&& (pixd_col<width)
    ){ // cache misses should occur whilst changing rows
      population_images[id
                        ]*=(1-opacity);
      population_images[id
                        ]+=opacity*g;
      
      pixd+=blockDim.x;
      pixd_row=pixd/width;
      pixd_col=pixd%width;
      id =rectangle_start +pixd_row*img_y +pixd_col;
     // __syncthreads(); 
    }
    //ensure previous rectangle has been drawn
    __syncthreads(); 
    mate_rectangle_idx+=genotype_length;
    __syncthreads(); 
  }
}
