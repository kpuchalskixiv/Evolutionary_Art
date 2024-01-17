#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>
#include "opencv2/core.hpp"

////////////////////////////////////////////////////////////////////////
// define kernel block size
////////////////////////////////////////////////////////////////////////

#define pop_size 64  // blockX
#define mate_size 16 // blockX
#define circles_per_thread 1 // blockZ

////////////////////////////////////////////////////////////////////////
// include kernel function
////////////////////////////////////////////////////////////////////////

#include <ES_kernel.h>

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////
float random_circle(
float max_radius_scale=1.0, float max_opacity_scale=1.0,
float x_min=0, float x_max=1, 
float y_min=0, float y_max=1)
{
  return 0
}

void save_best(){

}

float sum_array(float* arr, int start, int end){
  sum=0
  for(int i=start; i<end; i++)
    sum+=arr[i]
  return sum
}


int main(int argc, const char **argv){

  int children_per_mate=8, parents=8;
  int iters=20000, log_every=300;
  bool gray=true;
  int circle_dim;
  int img_x, img_y;


  long long bx, by, bz, i, j, k, ind;
  float    *h_u1, *h_u2, *h_foo,
           *d_u1, *d_u2, *d_foo;

  float *h_population,
        *d_population,
        *d_eval_population;
  

  if(gray) circle_dim=5;
  else circle_dim=7;

  size_t    bytes = (pop_size + children_per_mate*parents) *mate_size*circle_dim;
  size_t    eval_bytes =  (pop_size + children_per_mate*parents) *img_x*img_y;

  // grid is popsizeXcircler_per_mateXcricledim (x,y,r,alpha,gray - cause black&white for start)
  printf("Grid dimensions: %d x %d x %d \n\n", pop_size, mate_size, circle_dim);

  // allocate memory for arrays
  h_population = (float *)malloc(sizeof(float) * bytes);
  checkCudaErrors( cudaMalloc((void **)&d_population, sizeof(float) * bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_eval_population, sizeof(float) * eval_bytes) );


  // randomly initialize population
  
  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  //checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
  checkCudaErrors( curandGenerateUniform(gen, d_population, pop_size * mate_size * circle_dim) );

  // initialise card
  findCudaDevice(argc, argv);
  // initialise CUDA timing
  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  cudaEventRecord(start);


  // Set up the execution configuration
  int no_circles=4, update_frequency=100, last_update_iter=0;
  float *log_objective_values, *iter_evals;
  float add_cricle_threshold=0.01;
  log_objective_values=malloc(sizeof(float)*iters);

  int circles_to_mutate=pop_size*mat;
  //run evaluation on basic population
  int iter;
  for (iter=0; iter<iters; iter++) {
    // run parent selection

    // run crossover kernel
    // TODO

    // run mutation kernel, circle per thread
    //<<<nblocks,nthreads>>>
    mutate_kernel<<parents,mate_size*children_per_mate>>(d_population, parent_idxs, no_circles, children_per_mate) // add sigmas
    
    // run evaluation kerenel
    // probably most time consuming part so focus on pararellizing this
    evaluate_kernel<<parents,children_per_mate>>(d_population, d_eval_population)
    // select next population, little data  128~=pop_size+no_children floats

    //add synchThreads

    if (iter%log_every==0)
      save_best()

    if(iter%update_frequency==0 
      && iter-last_update_iter>=update_frequency*2 
      && no_circles<=mate_size){
      prev=sum_array(log_objective_values, no_iter-2*update_frequency, no_iter-update_frequency)
      curr=sum_array(log_objective_values, no_iter-update_frequency, no_iter)

      if((prev-curr)/prev < add_cricle_threshold*powf(0.995, no_circles)){
        no_circles+=1;
      }
      
    }
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("%dx ES CUDA time: %.1f (ms) \n\n", iter, milli);
    
 // Release GPU and CPU memory

  checkCudaErrors( cudaFree(d_population) );
  checkCudaErrors( cudaFree(d_eval_population) );
  free(h_population);
  free(log_objective_values);

  cudaDeviceReset();
}