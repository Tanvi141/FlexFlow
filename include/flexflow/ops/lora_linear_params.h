#ifndef _FLEXFLOW_LORA_LINEAR_PARAMS_H
#define _FLEXFLOW_LORA_LINEAR_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

class LoraLinearParams {
public:
  LayerID layer_guid;

  bool is_valid(std::pair<ParallelTensorShape, ParallelTensorShape> const
                    &input_shape) const;
  friend bool operator==(LoraLinearParams const &lhs,
                         LoraLinearParams const &rhs);
};

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::LoraLinearParams> {
  size_t operator()(FlexFlow::LoraLinearParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_LORA_LINEAR_PARAMS_H
