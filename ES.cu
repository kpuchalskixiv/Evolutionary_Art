#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <curand.h>
#include <cuda.h>

#include <helper_cuda.h>
#include <ES_kernel.h>


// utils



int cmp(const void *a, const void *b)
{
    struct loss_item *a1 = (struct loss_item *)a;
    struct loss_item *a2 = (struct loss_item *)b;
    if ((*a1).value < (*a2).value)
        return -1;
    else if ((*a1).value > (*a2).value)
        return 1;
    else
        return 0;
}
float sum_array(float* arr, int start, int end){
  int sum=0;
  for(int i=start; i<end; i++)
    sum+=arr[i];
  return sum;
}

void save_best(){

}



////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){

  int iters=20000, log_every=100;
  bool gray=true;
  int genotype_length;
  int img_x=256, img_y=256;

  float *best_mate_img,
        *d_population,
        *d_population_copy,
        *d_population_images,
        *d_mutation_coef,
        *d_target_img, *h_target_img;
  

  //if(gray) genotype_length=6;
  //else
  genotype_length=8;

  size_t    bytes = (pop_size + children_per_mate*parents) *mate_size*genotype_length;
  size_t    eval_bytes =  (pop_size + children_per_mate*parents) *img_x*img_y;

  // grid is popsizeXcircler_per_mateXcricledim (x1,y1,x2,y2,alpha,gray - cause black&white for start)
  printf("Grid dimensions: %d x %d x %d \n\n", pop_size, mate_size, genotype_length);
  // initialise card
  findCudaDevice(argc, argv);

  // allocate memory for arrays
  best_mate_img = (float *)malloc(sizeof(float) * img_y*img_x);
  h_target_img = (float *)malloc(sizeof(float) * img_y*img_x);
  checkCudaErrors( cudaMalloc((void **)&d_population, sizeof(float) * bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_population_copy, sizeof(float) * bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_population_images, sizeof(float) * eval_bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_mutation_coef, sizeof(float) *bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_target_img, sizeof(float)  *img_x*img_y) );

  for(int i=0; i<img_x*img_y; i++) h_target_img[i]=(float)rand()/(float)(RAND_MAX);
  checkCudaErrors(cudaMemcpy(d_target_img, h_target_img, //destination, source
                              img_x*img_y*sizeof(float),
                              cudaMemcpyHostToDevice) );

  // randomly initialize population
  
  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
  checkCudaErrors( curandGenerateUniform(gen, d_population, bytes) );
  checkCudaErrors( curandGenerateNormal(gen, d_mutation_coef, 
                                        bytes,
                                        1.0f, 0.001f) );


  // Set up the execution configuration
  int no_circles=4, update_frequency=100, last_update_iter=0;
  float *log_objective_values, *iter_evals;
  float add_cricle_threshold=0.01, prev, curr;
  log_objective_values=(float *)malloc(sizeof(float)*iters);

  int iter;
  for (iter=1; iter<=iters; iter++) {
    if (iter%log_every==0){
      save_best();
      printf("iter number: %d. Best MSE: %f \n", iter, population_losses[0].value);
 /*    printf("%f %d, %f %d, %f %d \n", 
      population_losses[0].value, population_losses[0].index,
      population_losses[pop_size-1].value, population_losses[pop_size-1].index,
      population_losses[pop_size+children_per_mate*parents-1].value, population_losses[pop_size+children_per_mate*parents-1].index
      );*/ 

    }
    
    // run selection, choose best parents and copy paste them into latter half of population array
    checkCudaErrors(cudaMemcpy(d_population_copy, d_population, //destination, source
                            bytes*sizeof(float),
                            cudaMemcpyDeviceToDevice) );
    cudaDeviceSynchronize();  
    selection_kernel<<<parents, genotype_length*mate_size>>>(d_population, d_population_copy, genotype_length);
    getLastCudaError("Selection kernel failed\n");
    cudaDeviceSynchronize();  
    checkCudaErrors(cudaMemcpy(d_population, d_population_copy, //destination, source
                            bytes*sizeof(float),
                            cudaMemcpyDeviceToDevice) );
    cudaDeviceSynchronize();  

    //run mutation
    checkCudaErrors( curandGenerateNormal(gen, d_mutation_coef, 
                                    (pop_size+children_per_mate*parents) * mate_size * genotype_length,
                                    1.0f, 0.001f) );
    cudaDeviceSynchronize();

    mutation_kernel<<<pop_size+children_per_mate*parents, genotype_length*mate_size>>>(d_population, d_mutation_coef, genotype_length);
    getLastCudaError("Mutation kernel failed\n");
    cudaDeviceSynchronize();    


    //run evaluation
    reset_images_kernel<<<pop_size+children_per_mate*parents, 512>>>(d_population_images, img_x, img_y);
    getLastCudaError("Reset kernel failed\n");
    cudaDeviceSynchronize();

    draw_kernel<<<pop_size+children_per_mate*parents, 512>>>(d_population, d_population_images, mate_size, img_x, img_y, genotype_length);
    getLastCudaError("Draw kernel failed\n");
    cudaDeviceSynchronize();

    eval_kernel<<<pop_size+children_per_mate*parents, 512, 512*sizeof(float)>>>(d_population_images, 
                                   d_target_img, img_x, img_y);
    getLastCudaError("Eval kernel failed\n");
    cudaDeviceSynchronize();

    qsort(population_losses, pop_size+children_per_mate*parents, sizeof(population_losses[0]), cmp);



    if(iter%update_frequency==0 
      && iter-last_update_iter>=update_frequency*2 
      && no_circles<=mate_size){
      prev=sum_array(log_objective_values, iter-2*update_frequency, iter-update_frequency);
      curr=sum_array(log_objective_values, iter-update_frequency, iter);

      if((prev-curr)/prev < add_cricle_threshold*powf(0.995, no_circles)){
        no_circles+=1;
      }
      
    }
  }

 // Release GPU and CPU memory

  checkCudaErrors( cudaFree(d_population) );
  checkCudaErrors( cudaFree(d_population_images) );
  checkCudaErrors( cudaFree(d_mutation_coef) );

  free(best_mate_img);
  free(log_objective_values);

  cudaDeviceReset();
}