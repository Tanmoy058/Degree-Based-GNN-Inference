/**
 *  Copyright (c) 2019-2022 by Contributors
 * @file array/uvm_array_op.h
 * @brief Array operator templates
 */
#ifndef DGL_ARRAY_UVM_ARRAY_OP_H_
#define DGL_ARRAY_UVM_ARRAY_OP_H_

#include <dgl/array.h>

#include <utility>

namespace dgl {
namespace aten {
namespace impl {

template <typename DType, typename IdType>
void IndexSelectInPlace(NDArray src, NDArray dst, IdArray src_rows, IdArray dst_rows);

// Take CPU array and GPU index, and then index with GPU.
template <typename DType, typename IdType>
NDArray Sample(NDArray U, IdArray indptr, IdArray rows, IdArray ret, IdArray num_threads);

// Take CPU array and GPU index, and then index with GPU.
template <typename DType, typename IdType>
NDArray Gather2D(NDArray array, IdArray index, IdArray num_threads);

// Take CPU array and GPU index, and then index with GPU.
template <typename DType, typename IdType>
NDArray IndexSelectCPUFromGPU(NDArray array, IdArray index);

template <typename DType, typename IdType>
void IndexScatterGPUToCPU(NDArray dest, IdArray index, NDArray source);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_UVM_ARRAY_OP_H_
