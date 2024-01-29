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
  float sum=0;
  for(int i=start; i<end; i++)
    sum+=arr[i];
  return sum;
}
float sum_mse(loss_item* arr, int start, int end){
  float sum=0;
  for(int i=start; i<end; i++)
    sum+=arr[i].value;
  return sum;
}

void save_best(float* d_best_img, float* h_best_img,
               float* d_best_mate, float* h_best_mate,
              int img_x, int img_y, int iter, int limit,
              float best_mse, float avg_mse){
  checkCudaErrors(cudaMemcpy(h_best_img, d_best_img, //destination, source
                              img_x*img_y*sizeof(float),
                              cudaMemcpyDeviceToHost) );
  checkCudaErrors(cudaMemcpy(h_best_mate, d_best_mate, //destination, source
                              mate_size*genotype_length*sizeof(float),
                              cudaMemcpyDeviceToHost) );  //save to file
  FILE *fp;
  if(iter==1) fp = fopen("../cuda_output/mona_results.txt", "w");
  else fp = fopen("../cuda_output/mona_results.txt", "a");
  // check for error here

  fprintf(fp, "\n %d, ", iter);
  for (unsigned i = 0; i < img_x*img_y; i++) {
      fprintf(fp, "%f, ", h_best_img[i]);
      // check for error here too
  }

  fclose(fp);

  if(iter==1){
    fp = fopen("../cuda_output/mona_mates.txt", "w");
   // fprintf("iter, no_squares, best_mse, avg_mse, %d", 0);
  }
  else fp = fopen("../cuda_output/mona_mates.txt", "a");
  // check for error here

  fprintf(fp, "\n %d, %d, %f, %f, ", iter, limit, best_mse, avg_mse);
  for (unsigned i = 0; i < mate_size*genotype_length; i++) {
      fprintf(fp, "%f, ", h_best_mate[i]);
      // check for error here too
  }

  fclose(fp);
}

void read_target(float* target_img,  int img_x, int img_y){
  FILE *fp;
  fp=fopen("../input/mona_gray_cuda.txt", "r");
  for(int i=0; i<img_x*img_y; i++){
    fscanf(fp, "%f", &target_img[i]);
  }
  fclose(fp);
}

void run_single_evaluation(
  float* d_population, 
  float* d_population_images, 
  float* d_target_img,
  int img_x, int img_y, int no_figures, int max_radius){
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
}
////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){

 // bool gray=true;
  int img_x=256, img_y=171;
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
  //size_t    sigmas_bytes =  (pop_size + children_per_mate*parents)*genotype_length;

  // grid is popsizeXcircler_per_mateXcricledim (x1,y1,x2,y2,alpha,gray - cause black&white for start)
  // initialise card
  findCudaDevice(argc, argv);

  // allocate memory for arrays
  best_mate_img = (float *)malloc(sizeof(float) * img_y*img_x);
  h_target_img = (float *)malloc(sizeof(float) * img_y*img_x);
  h_best_mate = (float *)malloc(sizeof(float) * mate_size*genotype_length);

  read_target(h_target_img, img_x, img_y);

  checkCudaErrors( cudaMalloc((void **)&d_population, sizeof(float) * bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_mutation_mates_coef, sizeof(float) * bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_mutation_mates_if, sizeof(float) * bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_population_copy, sizeof(float) * bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_population_images, sizeof(float) * eval_bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_target_img, sizeof(float)  *img_x*img_y) );
  
  checkCudaErrors( cudaMalloc((void **)&d_mutation_sigmas_coef, sizeof(float) *bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_mutation_sigmas_if, sizeof(float) *bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_sigmas, sizeof(float)* bytes) );

  reset_values_kernel<<<pop_size+children_per_mate*parents, mate_size*genotype_length>>>(d_sigmas, 0.05f, genotype_length);
  getLastCudaError("Reset kernel failed\n");
  checkCudaErrors( cudaMalloc((void **)&d_sigmas_copy, sizeof(float) *bytes) );


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
  //fix_param_kernel<<<pop_size+children_per_mate*parents, mate_size>>>(d_population, 3, 0.5f) ;
  //getLastCudaError("fixing opacity kernel failed\n");
 // cudaDeviceSynchronize();  

 
  // Set up the execution configuration
  int iters=1000000, log_every=10000,

  no_figures=4, update_frequency=1000;


  
  int last_update_iter=0;
  float *log_objective_values;
  float add_cricle_threshold=0.005, prev, curr, max_mse, r_float, avg_mse;
  log_objective_values=(float *)malloc(sizeof(float)*iters);
  int iter, m=0;

  run_single_evaluation(d_population, 
                        d_population_images, 
                        d_target_img,
                        img_x,  img_y,  no_figures,  max_radius);
                        
  for (iter=0; iter<=iters; iter++) {
    if ((iter%log_every==0) || (iter==iters)){
      
      avg_mse=sum_mse(population_losses, 0, pop_size)/(float)pop_size;
      save_best(d_population_images+population_losses[0].index*img_x*img_y, 
                best_mate_img, 
                d_population+population_losses[0].index*mate_size*genotype_length,
                h_best_mate,
                img_x, img_y, iter, no_figures, 
                population_losses[0].value, avg_mse);
      printf("iter number: %d. Best MSE: %f \n", iter, population_losses[0].value);
      printf("Current #rectangles is %d\n", no_figures);
    }
    
    // run selection, choose best parents and copy paste them into latter half of population array
    max_mse=population_losses[pop_size-1].value;
    fitness_values_kernel<<<1, pop_size, sizeof(float)*pop_size>>>(max_mse);
    getLastCudaError("fitness values kernel failed\n");
    cudaDeviceSynchronize();  
    for(int p=0; p<parents;p++){
      r_float=(float)rand()/(float)(RAND_MAX);
      m=0;
      while((r_float>=0) && (m<pop_size)){
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
                        bytes*sizeof(float),
                        cudaMemcpyDeviceToDevice) );
    cudaDeviceSynchronize();  

    population_selection_kernel<<<parents, genotype_length*mate_size>>>(d_population, d_population_copy);
    getLastCudaError("Population children kernel failed\n");
    population_selection_kernel<<<parents, genotype_length*mate_size>>>(d_sigmas, d_sigmas_copy);
    getLastCudaError("Population children kernel failed\n");
    cudaDeviceSynchronize();  

    create_children_kernel<<<parents, genotype_length*mate_size>>>(d_population, d_population_copy);
    getLastCudaError("Population children kernel failed\n");
    create_children_kernel<<<parents, genotype_length*mate_size>>>(d_sigmas, d_sigmas_copy);
//    sigmas_selection_kernel<<<parents, genotype_length>>>(d_sigmas, d_sigmas_copy);
    getLastCudaError("Sigmas children kernel failed\n");
    cudaDeviceSynchronize();  

    checkCudaErrors(cudaMemcpy(d_population, d_population_copy, //destination, source
                            bytes*sizeof(float),
                            cudaMemcpyDeviceToDevice) );
    checkCudaErrors(cudaMemcpy(d_sigmas, d_sigmas_copy, //destination, source
                            bytes*sizeof(float),
                            cudaMemcpyDeviceToDevice) );
    cudaDeviceSynchronize();  

    //run mutation
    checkCudaErrors( curandGenerateNormal(gen, d_mutation_sigmas_coef, 
                                    bytes,
                                    0.0f, 0.001f) );
    checkCudaErrors( curandGenerateUniform(gen, d_mutation_sigmas_if, bytes));
    checkCudaErrors( curandGenerateUniform(gen, d_mutation_mates_if, bytes));
    checkCudaErrors( curandGenerateUniform(gen, d_mutation_mates_coef, bytes));
    cudaDeviceSynchronize();


//    sigmas_mutation_kernel<<<children_per_mate*parents, genotype_length>>>(d_sigmas,  d_mutation_sigmas_coef, d_mutation_sigmas_if, true);
    sigmas_mutation_kernel<<<children_per_mate*parents, no_figures*genotype_length>>>(d_sigmas,  d_mutation_sigmas_coef, d_mutation_sigmas_if);
    getLastCudaError("Sigmas mutation kernel failed\n");
    cudaDeviceSynchronize();
    mate_mutation_kernel<<<children_per_mate*parents, genotype_length*no_figures>>>(d_population, d_mutation_mates_coef, d_mutation_mates_if, d_sigmas);
    getLastCudaError("Population mutation kernel failed\n");
    cudaDeviceSynchronize();    

    //run evaluation
    run_single_evaluation(d_population, 
                          d_population_images, 
                          d_target_img,
                          img_x,  img_y,  no_figures,  max_radius);


    log_objective_values[iter]=population_losses[0].value;
    if(iter%update_frequency==0 
      && iter-last_update_iter>=update_frequency*2 
      && no_figures<mate_size){
      prev=sum_array(log_objective_values, iter-2*update_frequency, iter-update_frequency);
      curr=sum_array(log_objective_values, iter-update_frequency, iter);

      if((prev-curr)/prev < add_cricle_threshold*powf(0.995, no_figures)){
        last_update_iter=iter;
        no_figures+=1;
       // printf("Added rectangle at iter %d, current # is %d\n", iter, no_figures);
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