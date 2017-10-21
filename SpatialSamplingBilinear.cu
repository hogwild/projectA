#include "luaT.h"
#include "THC.h"
#include "utils.h"

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

//#define CUDA_MAX_THREADS 256 

/*
 * Description:
 */

__device__ float get_samplevalue(int ind, int d1, int d2, int d3, float wratio, float hratio,float *input)
{
  //ind2sub:a d1(z)*d2(y)*d3(x) array
  int x,y,z ;//,"v" note: replace v by x to reduce the number of variable
  x = (ind)%(d2*d3); //note:ind start from 0.
  z = (ind-x)/(d2*d3);
  ind = x;
  x = (ind)%d3;
  y = (ind-x)/d3;
 //subpixel position
  float ix = x*wratio;
  float iy = y*hratio;
 //4 nearest neighbors
  float ix_nw = floor(ix);
  float iy_nw = floor(iy);
  float ix_ne = ix_nw + 1;
  float iy_ne = iy_nw;
  float ix_sw = ix_nw;
  float iy_sw = iy_nw + 1;
  float ix_se = ix_nw + 1;
  float iy_se = iy_nw + 1;
  // get surfaces to each neighbor:
  float se = (ix-ix_nw)*(iy-iy_nw);
  float sw = (ix_ne-ix)*(iy-iy_ne);
  float ne = (ix-ix_sw)*(iy_sw-iy);
  float nw = (ix_se-ix)*(iy_se-iy);
 //sub2ind: subpixel position
  d2 = 1+(d2-1)*hratio; // update the d2 to be the d2 of input.
  d3 = 1+(d3-1)*wratio; // update the d3 to be the d3 of input.
  int inw = ix_nw+(iy_nw+z*d2)*d3; //note: inw is an int.
  int ine = min((int)ix_ne,d3-1)+(iy_ne+z*d2)*d3;
  int isw = ix_sw+(min((int)iy_sw,d2-1)+z*d2)*d3;
  int ise = min((int)ix_se,d3-1)+(min((int)iy_se,d2-1)+z*d2)*d3;
 //compute the sample value 
  float sum = input[inw]*nw+input[ine]*ne+input[isw]*sw+input[ise]*se;

 // float sum = THCudaTensor_get3d(state,input,z,iy_nw,ix_nw)*nw+THCudaTensor_get3d(state,input,z,iy_ne,min((int)ix_ne,d3-1))*ne+THCudaTensor_get3d(state,input,z,min((int)iy_sw,d3-1),ix_sw)*sw+THCudaTensor_get3d(state,input,z,min((int)iy_se,d2-1),min((int)ix_se,d3-1))*se;
 
  return sum;
}

/*__device__ void get_samplevalue_inv(int ind, int d1, int d2, int d3, float wratio, float hratio,float *gradInput, float *gradOutput)
{
 //ind2sub:a d1(z)*d2(y)*d3(x) array
  int x,y,z ;//,"v" note: replace v by x to reduce the number of variable
  x = (ind)%(d2*d3); //note:ind start from 0.
  z = (ind-x)/(d2*d3);
  ind = x;
  x = (ind)%d3;
  y = (ind-x)/d3;
 //subpixel position
  float ix = x*wratio;
  float iy = y*hratio;
 //4 nearest neighbors
  float ix_nw = floor(ix);
  float iy_nw = floor(iy);
  float ix_ne = ix_nw + 1;
  float iy_ne = iy_nw;
  float ix_sw = ix_nw;
  float iy_sw = iy_nw + 1;
  float ix_se = ix_nw + 1;
  float iy_se = iy_nw + 1;
  // get surfaces to each neighbor:
  float se = (ix-ix_nw)*(iy-iy_nw);
  float sw = (ix_ne-ix)*(iy-iy_ne);
  float ne = (ix-ix_sw)*(iy_sw-iy);
  float nw = (ix_se-ix)*(iy_se-iy);
 //sub2ind: subpixel position
  d2 = 1+(d2-1)*hratio; // update the d2 to be the d2 of input.
  d3 = 1+(d3-1)*wratio; // update the d3 to be the d3 of input.
  int inw = ix_nw+(iy_nw+z*d2)*d3; //note: inw is an int.
  int ine = min((int)ix_ne,d3-1)+(iy_ne+z*d2)*d3;
  int isw = ix_sw+(min((int)iy_sw,d2-1)+z*d2)*d3;
  int ise = min((int)ix_se,d3-1)+(min((int)iy_se,d2-1)+z*d2)*d3;
 
 // thread ID
 // int tid = ind%CUDA_MAX_THREADS;

 //create array to hold partial sums. Note: we should avoid the race condition.
 //__shared__ float sum[CUDA_MAX_THREADS][4]; // note: '256' is the number of thread in a block 
  //sum[tid][0]=0;
  //sum[tid][1]=0;
  //sum[tid][2]=0;
  //sum[tid][3]=0;
  //sum[tid][0] += gradOutput[ind]*nw;
  //sum[tid][1] += gradOutput[ind]*ne;
  //sum[tid][2] += gradOutput[ind]*sw;
  //sum[tid][3] += gradOutput[ind]*se;
  //__syncthreads();

 //update the gradInput by the value from gradOutput. Note: we should avoid the race condition.
 //atomicAdd(&gradInput[inw],gradOutput[ind]*nw);
 //atomicAdd(&gradInput[ine],gradOutput[ind]*ne);
 //atomicAdd(&gradInput[isw],gradOutput[ind]*sw);
 //atomicAdd(&gradInput[ise],gradOutput[ind]*se);
  gradInput[inw] += gradOutput[ind]*nw;
  gradInput[ine] += gradOutput[ind]*ne;
  gradInput[isw] += gradOutput[ind]*sw;
  gradInput[ise] += gradOutput[ind]*se;
  //__syncthreads();
  return;

}*/

__global__ void rescale(float *input, float *output, long no_oelements,
                        float wratio, float hratio, int d1, int d2, int d3)
{
  // output offset:
  long ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_oelements) return; 
  
   output[ii] = get_samplevalue(ii, d1, d2, d3, wratio, hratio, input);
}


static int cunn_SpatialSamplingBilinear_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  //THCudaTensor_zero(state, output);
  int owidth = luaT_getfieldcheckint(L, 1, "owidth");
  int oheight = luaT_getfieldcheckint(L,1,"oheight");

  THAssert(THCudaTensor_checkGPU(state, 2, input, output));

  input = THCudaTensor_newContiguous(state, input);
  float *input_data = THCudaTensor_data(state, input);

  THCudaTensor_resize3d(state,output,input->size[0],oheight,owidth); 
  //THCudaTensor_zero(state, output); //for test purpose
  float *output_data = THCudaTensor_data(state, output);

  // This is for allocating output Tensor
  long no_elements = max((float)(owidth*oheight*(input->size[0])),(float)((input->size[0])*(input->size[1])*(input->size[2]))); // the number of elements should be the max of the input and output.
  long no_oelements = owidth*oheight*(input->size[0]); // the numbe of output elements.
  int d1 = input->size[0];//the number of channels
  int d2 = oheight; // the height of the output
  int d3 = owidth; // the width of the output
  float wratio = (float)((input->size[2])-1)/(owidth-1);
  float hratio = (float)((input->size[1])-1)/(oheight-1); 
  
  // cuda blocks & threads:
  long nthreads = 256;
  // Max number of blocks: http://en.wikipedia.org/wiki/CUDA
  // 65535 for SM 2.x, 2^32 -1 for >= 3.0
  // TODO: When we move to SM 3.5 we should update this
  //long n_xblocks = max(ceil((float)owidht/nthreads),1);
  //long n_yblocks = max(ceil((float)oheight*d1),1);
  long n_xblocks = min(max((int)ceil((float)no_elements/nthreads), 1), 65535);
  long n_yblocks = (long)ceil((float)no_elements/(float)(n_xblocks*nthreads));
  if (n_yblocks > 65535) {
    THError("Input size is too large!  aborting");
  }
  dim3 blocks(n_xblocks, n_yblocks);
  dim3 threads(nthreads);

  // kernel:
  rescale<<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (input_data, output_data, no_oelements, wratio, hratio, d1, d2, d3); // 0 is the device number, means GPU No.1, THCState_get.. is the process that this kernel will be put into.

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialUpSamplingNearest.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }

  // clean:
  THCudaTensor_free(state, input);

  return 1;
}

/*
 * Description:
 */
__global__ void rescale_inv(float *gradInput_data, float *gradOutput_data, long no_oelements,
                              float wratio, float hratio, int d1, int d2, int d3)
{
  // output offset:
  long ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_oelements) return;
  //get_samplevalue_inv(ii, d1, d2, d3, wratio, hratio, gradInput_data,gradOutput_data);
  //return;
  //ind2sub:a d1(z)*d2(y)*d3(x) array
  int ind =ii;
  int x,y,z ;//,"v" note: replace v by x to reduce the number of variable
  x = (ind)%(d2*d3); //note:ind start from 0.
  z = (ind-x)/(d2*d3);
  ind = x;
  x = (ind)%d3;
  y = (ind-x)/d3;
 //subpixel position
  float ix = x*wratio;
  float iy = y*hratio;
 //4 nearest neighbors
  float ix_nw = floor(ix);
  float iy_nw = floor(iy);
  float ix_ne = ix_nw + 1;
  float iy_ne = iy_nw;
  float ix_sw = ix_nw;
  float iy_sw = iy_nw + 1;
  float ix_se = ix_nw + 1;
  float iy_se = iy_nw + 1;
  // get surfaces to each neighbor:
  float se = (ix-ix_nw)*(iy-iy_nw);
  float sw = (ix_ne-ix)*(iy-iy_ne);
  float ne = (ix-ix_sw)*(iy_sw-iy);
  float nw = (ix_se-ix)*(iy_se-iy);
 //sub2ind: subpixel position
  d2 = 1+(d2-1)*hratio; // update the d2 to be the d2 of input.
  d3 = 1+(d3-1)*wratio; // update the d3 to be the d3 of input.
  int inw = ix_nw+(iy_nw+z*d2)*d3; //note: inw is an int.
  int ine = min((int)ix_ne,d3-1)+(iy_ne+z*d2)*d3;
  int isw = ix_sw+(min((int)iy_sw,d2-1)+z*d2)*d3;
  int ise = min((int)ix_se,d3-1)+(min((int)iy_se,d2-1)+z*d2)*d3;

 //update the gradInput by the value from gradOutput. Note: we should avoid the race condition.
 atomicAdd(&gradInput_data[inw],gradOutput_data[ii]*nw);
 atomicAdd(&gradInput_data[ine],gradOutput_data[ii]*ne);
 atomicAdd(&gradInput_data[isw],gradOutput_data[ii]*sw);
 atomicAdd(&gradInput_data[ise],gradOutput_data[ii]*se);
  //gradInput[inw] += gradOutput[ind]*nw;
  //gradInput[ine] += gradOutput[ind]*ne;
  //gradInput[isw] += gradOutput[ind]*sw;
  //gradInput[ise] += gradOutput[ind]*se;
  //__syncthreads();
}


static int cunn_SpatialSamplingBilinear_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");  
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 3, input, gradOutput, gradInput));
  
  float *gradOutput_data = THCudaTensor_data(state, gradOutput);
  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_zero(state, gradInput);
  float *gradInput_data = THCudaTensor_data(state, gradInput);
  

// dim
  int channelDim = 0;
  int iwidth = input->size[channelDim+2];
  int iheight = input->size[channelDim+1];
  int ichannels = input->size[channelDim];
  int owidth = gradOutput->size[channelDim+2];
  int oheight = gradOutput->size[channelDim+1];
  int ochannels = gradOutput->size[channelDim];

  long no_elements = max(iwidth*iheight*ichannels,owidth*oheight*ochannels);
  long no_oelements = owidth*oheight*ochannels;

  int d1;
  int d2;
  int d3;

  if (gradInput->nDimension == 3) {
    d1 = gradOutput->size[0]; // the inverse process is still based on the gradOutput. Each point in gradOutput is relateded to 4 points in gradInput.
    d2 = gradOutput->size[1];
    d3 = gradOutput->size[2];
  } else {
    d1 = gradOutput->size[1]; // omit this 'else'. presently, the batch model is not contained.
    d2 = gradOutput->size[2];
    d3 = gradOutput->size[3];
  }
  
  float wratio = (float)(iwidth-1)/(owidth-1);
  float hratio = (float)(iheight-1)/(oheight-1);

  // cuda blocks & threads:
  long nthreads = 256;
  // Max number of blocks: http://en.wikipedia.org/wiki/CUDA
  // 65535 for SM 2.x, 2^32 -1 for >= 3.0
  // TODO: When we move to SM 3.5 we should update this
  long n_xblocks = min(max((int)ceil((float)no_elements/nthreads), 1), 65535);
  long n_yblocks = (long)ceil((float)no_elements/(float)(n_xblocks*nthreads));
  if (n_yblocks > 65535) {
    THError("Input size is too large!  aborting");
  }
  dim3 blocks(n_xblocks, n_yblocks);
  dim3 threads(nthreads);

  // kernel:
  rescale_inv<<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data, no_oelements,
   wratio, hratio, d1, d2, d3);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialSamplingBilinear.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  // cleanup
 
  //THCudaTensor_free(state,output);

  return 1;
}

static const struct luaL_Reg cunn_SpatialSamplingBilinear__ [] = {
  {"SpatialSamplingBilinear_updateOutput", cunn_SpatialSamplingBilinear_updateOutput},
  {"SpatialSamplingBilinear_updateGradInput", cunn_SpatialSamplingBilinear_updateGradInput},
  {NULL, NULL}
};

void cunn_SpatialSamplingBilinear_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialSamplingBilinear__, "nn");
  lua_pop(L,1);
}
