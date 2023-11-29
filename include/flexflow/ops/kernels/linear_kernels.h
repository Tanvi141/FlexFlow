#ifndef _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include <unordered_map>
#include <vector>

namespace FlexFlow {

class LinearMeta : public OpMeta {
public:
  LinearMeta(FFHandler handle, int batch_size);
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t outputTensor;
  cudnnActivationDescriptor_t actiDesc;
#else
  miopenTensorDescriptor_t outputTensor;
  miopenActivationDescriptor_t actiDesc;
#endif
  float const *one_ptr;
  ActiMode activation;
  RegularizerMode kernel_reg_type;
  float kernel_reg_lambda;
  bool use_bias;
  DataType input_type, weight_type, output_type;
  char op_name[MAX_OPNAME];
};

using LinearFunctionType = std::function<void(LinearMeta const *,
                  void const *,
                  void *,
                  void const *,
                  void const *,
                  int,
                  int,
                  int,
                  ffStream_t)>;

namespace Kernels {
namespace Linear {

class LinearKernelSelector {
public:
  LinearFunctionType selectLinearForwardKernel(int in_dim, int out_dim, int batch_size);
private:
  std::map<std::vector<int>, LinearFunctionType> cache;
};

void init_kernel(LinearMeta *m, int batch_size, int channel);
void forward_kernel_wrapper(LinearMeta const *m,
                            void const *input_ptr,
                            void *output_ptr,
                            void const *filter_ptr,
                            void const *bias_ptr,
                            int in_dim,
                            int out_dim,
                            int batch_size);
void backward_kernel_wrapper(LinearMeta const *m,
                             void const *input_ptr,
                             void *input_grad_ptr,
                             void const *output_ptr,
                             void *output_grad_ptr,
                             void const *kernel_ptr,
                             void *kernel_grad_ptr,
                             void *bias_ptr,
                             int in_dim,
                             int out_dim,
                             int batch_size);
bool use_activation(ActiMode mode);

namespace Internal {
void forward_kernel(LinearMeta const *m,
                    void const *input_ptr,
                    void *output_ptr,
                    void const *filter_ptr,
                    void const *bias_ptr,
                    int in_dim,
                    int out_dim,
                    int batch_size,
                    ffStream_t stream);
void backward_kernel(LinearMeta const *m,
                     void const *input_ptr,
                     void *input_grad_ptr,
                     void const *output_ptr,
                     void *output_grad_ptr,
                     void const *kernel_ptr,
                     void *kernel_grad_ptr,
                     void *bias_ptr,
                     int in_dim,
                     int out_dim,
                     int batch_size,
                     ffStream_t stream);
} // namespace Internal
} // namespace Linear
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H
