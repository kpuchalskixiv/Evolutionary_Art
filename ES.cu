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

void save_best(float* d_best_img, float* h_best_img,
               float* d_best_mate, float* h_best_mate,
              int img_x, int img_y, int iter, int limit){
  checkCudaErrors(cudaMemcpy(h_best_img, d_best_img, //destination, source
                              img_x*img_y*sizeof(float),
                              cudaMemcpyDeviceToHost) );
  checkCudaErrors(cudaMemcpy(h_best_mate, d_best_mate, //destination, source
                              mate_size*genotype_length*sizeof(float),
                              cudaMemcpyDeviceToHost) );  //save to file
  FILE *fp;
  if(iter==1) fp = fopen("./cuda_output/mona_results.txt", "w");
  else fp = fopen("./cuda_output/mona_results.txt", "a");
  // check for error here

  fprintf(fp, "\n %d, ", iter);
  for (unsigned i = 0; i < img_x*img_y; i++) {
      fprintf(fp, "%f, ", h_best_img[i]);
      // check for error here too
  }

  fclose(fp);

  if(iter==1) fp = fopen("./cuda_output/mona_mates.txt", "w");
  else fp = fopen("./cuda_output/mona_mates.txt", "a");
  // check for error here

  fprintf(fp, "\n %d, %d, ", iter, limit);
  for (unsigned i = 0; i < mate_size*genotype_length; i++) {
      fprintf(fp, "%f, ", h_best_mate[i]);
      // check for error here too
  }

  fclose(fp);
}

void read_target(float* target_img,  int img_x, int img_y){
  FILE *fp;
  fp=fopen("./input/mona_gray_cuda.txt", "r");
  for(int i=0; i<img_x*img_y; i++){
    fscanf(fp, "%f", &target_img[i]);
  }
  fclose(fp);
}


////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){

 // bool gray=true;
  int img_x=381, img_y=256;
  float max_radius=sqrtf(powf(img_x,2)+powf(img_y, 2))/2;

  float *best_mate_img, *h_best_mate,
        *d_population,
        *d_population_copy,
        *d_population_images,
        *d_mutation_mates_coef, *d_mutation_mates_if,
        *d_sigmas, *d_sigmas_copy,
        *d_mutation_sigmas_coef, *d_mutation_sigmas_if,
        *d_target_img, *h_target_img;

  size_t    bytes = (pop_size + children_per_mate*parents) *mate_size*genotype_length;
  size_t    eval_bytes =  (pop_size + children_per_mate*parents) *img_x*img_y;
  size_t    sigmas_bytes =  (pop_size + children_per_mate*parents)*genotype_length;

  // grid is popsizeXcircler_per_mateXcricledim (x1,y1,x2,y2,alpha,gray - cause black&white for start)
  printf("Grid dimensions: %d x %d x %d \n\n", pop_size, mate_size, genotype_length);
  // initialise card
  findCudaDevice(argc, argv);

  // allocate memory for arrays
  best_mate_img = (float *)malloc(sizeof(float) * img_y*img_x);
  h_target_img = (float *)malloc(sizeof(float) * img_y*img_x);
  h_best_mate = (float *)malloc(sizeof(float) * mate_size*genotype_length);

  read_target(h_target_img, img_x, img_y);
  printf("Target image read, sample pixel= %f\n", h_target_img[2137]);

  checkCudaErrors( cudaMalloc((void **)&d_population, sizeof(float) * bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_mutation_mates_coef, sizeof(float) * bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_mutation_mates_if, sizeof(float) * bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_population_copy, sizeof(float) * bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_population_images, sizeof(float) * eval_bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_target_img, sizeof(float)  *img_x*img_y) );
  
  checkCudaErrors( cudaMalloc((void **)&d_mutation_sigmas_coef, sizeof(float) *sigmas_bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_mutation_sigmas_if, sizeof(float) *sigmas_bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_sigmas, sizeof(float)* sigmas_bytes) );
  reset_values_kernel<<<pop_size+children_per_mate*parents, genotype_length>>>(d_sigmas, 0.05f, genotype_length);
  getLastCudaError("Reset kernel failed\n");
  checkCudaErrors( cudaMalloc((void **)&d_sigmas_copy, sizeof(float) *sigmas_bytes) );


  //for(int i=0; i<img_x*img_y; i++) h_target_img[i]=(float)rand()/(float)(RAND_MAX);
  checkCudaErrors(cudaMemcpy(d_target_img, h_target_img, //destination, source
                              img_x*img_y*sizeof(float),
                              cudaMemcpyHostToDevice) );

  // randomly initialize population
  
  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
 // checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
  checkCudaErrors( curandGenerateUniform(gen, d_population, bytes) );
 // checkCudaErrors( curandGenerateNormal(gen, d_population, bytes, 0.5f, 0.2f) );



  // Set up the execution configuration
  int iters=100000, log_every=1000,

  no_figures=1, update_frequency=2000;


  
  int last_update_iter=0;
  float *log_objective_values;
  float add_cricle_threshold=0.01, prev, curr, max_mse, r_float;
  log_objective_values=(float *)malloc(sizeof(float)*iters);
  int iter, m=0;

  for (iter=0; iter<=iters; iter++) {
    if ((iter%log_every==1) || (iter==iters)){
      save_best(d_population_images+population_losses[0].index*img_x*img_y, 
                best_mate_img, 
                d_population+population_losses[0].index*mate_size*genotype_length,
                h_best_mate,
                img_x, img_y, iter, no_figures);
      printf("iter number: %d. Best MSE: %f \n", iter, population_losses[0].value);
 /*    printf("%f %d, %f %d, %f %d \n", 
      population_losses[0].value, population_losses[0].index,
      population_losses[pop_size-1].value, population_losses[pop_size-1].index,
      population_losses[pop_size+children_per_mate*parents-1].value, population_losses[pop_size+children_per_mate*parents-1].index
      );*/ 

    }
    
    // run selection, choose best parents and copy paste them into latter half of population array
    max_mse=population_losses[pop_size+children_per_mate*parents-1].value;
    fitness_values_kernel<<<1, pop_size+children_per_mate*parents, sizeof(float)*pop_size*children_per_mate*parents>>>(max_mse);
    getLastCudaError("fitness values kernel failed\n");
    cudaDeviceSynchronize();  
    for(int p=0; p<parents;p++){
      r_float=(float)rand()/(float)(RAND_MAX);
      m=0;
      while(r_float>=0){
        r_float-=population_losses[m].value;
        m++;
      }
      parents_ids[p]=m-1;
    }
    /////////
    checkCudaErrors(cudaMemcpy(d_population_copy, d_population, //destination, source
                            bytes*sizeof(float),
                            cudaMemcpyDeviceToDevice) );
    checkCudaErrors(cudaMemcpy(d_sigmas_copy, d_sigmas, //destination, source
                        sigmas_bytes*sizeof(float),
                        cudaMemcpyDeviceToDevice) );
    cudaDeviceSynchronize();  
    population_selection_kernel<<<parents, genotype_length*mate_size>>>(d_population, d_population_copy);
    getLastCudaError("Population selection kernel failed\n");
    population_selection_kernel<<<parents, genotype_length>>>(d_sigmas, d_sigmas_copy);
    getLastCudaError("Sigmas selection kernel failed\n");
    cudaDeviceSynchronize();  
    checkCudaErrors(cudaMemcpy(d_population, d_population_copy, //destination, source
                            bytes*sizeof(float),
                            cudaMemcpyDeviceToDevice) );
    checkCudaErrors(cudaMemcpy(d_sigmas, d_sigmas_copy, //destination, source
                            sigmas_bytes*sizeof(float),
                            cudaMemcpyDeviceToDevice) );
    cudaDeviceSynchronize();  

    //run mutation
    checkCudaErrors( curandGenerateNormal(gen, d_mutation_sigmas_coef, 
                                    sigmas_bytes,
                                    0.0f, 0.001f) );
    checkCudaErrors( curandGenerateUniform(gen, d_mutation_sigmas_if, sigmas_bytes));
    checkCudaErrors( curandGenerateUniform(gen, d_mutation_mates_if, bytes));
    checkCudaErrors( curandGenerateUniform(gen, d_mutation_mates_coef, bytes));
    cudaDeviceSynchronize();


    sigmas_mutation_kernel<<<children_per_mate*parents, genotype_length>>>(d_sigmas,  d_mutation_sigmas_coef, d_mutation_sigmas_if, true);
    getLastCudaError("Sigmas mutation kernel failed\n");
    cudaDeviceSynchronize();
    mate_mutation_kernel<<<children_per_mate*parents, genotype_length*no_figures>>>(d_population, d_mutation_mates_coef, d_mutation_mates_if, d_sigmas, true);
    getLastCudaError("Population mutation kernel failed\n");
    cudaDeviceSynchronize();    





    //run evaluation
    //mem set doesnt work with floats
   // checkCudaErrors(cudaMemset(d_population_images, 0.0f, sizeof(float)*img_x*img_y*(pop_size+children_per_mate*parents)));

    reset_values_kernel<<<pop_size+children_per_mate*parents, 1024>>>(d_population_images, 0.0f, img_x*img_y);
    getLastCudaError("Reset kernel failed\n");
    cudaDeviceSynchronize();

    draw_kernel<<<pop_size+children_per_mate*parents, 1024>>>(d_population, d_population_images, no_figures, img_x, img_y, max_radius);
    getLastCudaError("Draw kernel failed\n");
    cudaDeviceSynchronize();

    eval_kernel<<<pop_size+children_per_mate*parents, 1024, 1024*sizeof(float)>>>(d_population_images, 
                                   d_target_img, img_x, img_y);
    getLastCudaError("Eval kernel failed\n");
    cudaDeviceSynchronize();

    qsort(population_losses, pop_size+children_per_mate*parents, sizeof(population_losses[0]), cmp);


    log_objective_values[iter]=population_losses[0].value;
    if(iter%update_frequency==0 
      && iter-last_update_iter>=update_frequency*2 
      && no_figures<mate_size){
      prev=sum_array(log_objective_values, iter-2*update_frequency, iter-update_frequency);
      curr=sum_array(log_objective_values, iter-update_frequency, iter);

      if((prev-curr)/prev < add_cricle_threshold*powf(0.995, no_figures)){
        last_update_iter=iter;
        no_figures+=1;
        printf("Added rectangle at iter %d, current # is %d\n", iter, no_figures);
      }
      
    }
  }

 // Release GPU and CPU memory

  checkCudaErrors( cudaFree(d_population) );
  checkCudaErrors( cudaFree(d_population_copy) );
  checkCudaErrors( cudaFree(d_population_images) );
  checkCudaErrors( cudaFree(d_mutation_mates_coef) );
  checkCudaErrors( cudaFree(d_mutation_mates_if) );
  checkCudaErrors( cudaFree(d_sigmas) );
  checkCudaErrors( cudaFree(d_sigmas_copy) );
  checkCudaErrors( cudaFree(d_mutation_sigmas_coef) );
  checkCudaErrors( cudaFree(d_mutation_sigmas_if) );
  checkCudaErrors( cudaFree(d_target_img) );

  free(best_mate_img);
  free(h_target_img);
  free(h_best_mate);
  free(log_objective_values);

  cudaDeviceReset();
}