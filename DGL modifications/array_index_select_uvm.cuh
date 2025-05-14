/**
 *  Copyright (c) 2021 by Contributors
 * @file array/cpu/array_index_select_uvm.cuh
 * @brief Array index select GPU kernel implementation
 */

#ifndef DGL_ARRAY_CUDA_UVM_ARRAY_INDEX_SELECT_UVM_CUH_
#define DGL_ARRAY_CUDA_UVM_ARRAY_INDEX_SELECT_UVM_CUH_

#define CACHE_LINE_SIZE 128

namespace dgl {
namespace aten {
namespace impl {

/**
 *  This is a cross-device access version of IndexSelectMultiKernel.
 *  Since the memory access over PCIe is more sensitive to the
 *  data access aligment (cacheline), we need a separate version here.
 */
 
template <typename DType, typename IdType>
__global__ void IndexSelectInPlaceKernel(
    const DType* const src, DType* dst, const IdType* const src_rows,
    const IdType* const dst_rows, const int64_t num_feat, const int64_t len) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < len) {
    int64_t col = threadIdx.x;
    const int64_t in_row = src_rows[out_row_index];
    const int64_t idx_offset =
        ((uint64_t)(&src[in_row * num_feat]) % CACHE_LINE_SIZE) /
        sizeof(DType);
    col = col - idx_offset;
    const auto out_row = dst_rows[out_row_index];
    while (col < num_feat) {
      if (col >= 0)
        dst[out_row * num_feat + col] = src[in_row * num_feat + col];
      col += blockDim.x;
    }
    out_row_index += stride;
  }
} 




template <typename DType, typename IdType>
__global__ void IndexSelectNA(
    const DType* const array, const int64_t num_feat, const IdType* const index,
    const int64_t length, const int64_t arr_len, DType* const out,
    const int64_t* perm = nullptr) {
    
  const int64_t offset = threadIdx.x;
  int64_t out_row_index = blockIdx.x;
  const int64_t stride_h = blockDim.x;
  const int64_t stride_v = gridDim.x;
  
  while (out_row_index < length) {
    const int64_t in_row = index[out_row_index];
    const auto out_row = perm ? perm[out_row_index] : out_row_index;
    const int64_t base_read_addr = in_row * num_feat;
    const auto base_write_addr = out_row * num_feat;
    assert(in_row >= 0 && in_row < arr_len);
    for (int64_t i = offset; i < num_feat; i += stride_h) {
        out[base_write_addr + i] = array[base_read_addr + i];
    }
    out_row_index += stride_v;
  }
}



 
 
template <typename DType>
__global__ void SampleKernel(
    DType* U, int32_t* ret, int32_t* indptr, int32_t* indptr_out, const int64_t num, const int64_t len) {
  int64_t index = blockIdx.x;
  int64_t offset = threadIdx.x;
  int64_t stride = blockDim.x;
  int64_t stride1 = gridDim.x;
  while (index < num) {
    int64_t base_read_addr = indptr[index];
    int64_t base_write_addr = indptr_out[index];
    int64_t deg = indptr_out[index + 1] - indptr_out[index];
    for (int64_t i = offset; i < deg; i += stride) {
        ret[base_write_addr + i] = U[base_read_addr + i];
        ret[base_write_addr + i + len] = index;
    }
    index += stride1;
  }
}

template <typename DType>
__global__ void SampleKernel(
    DType* U, int64_t* ret, int64_t* indptr, int64_t* indptr_out, const int64_t num, const int64_t len) {
  int64_t index = blockIdx.x;
  int64_t offset = threadIdx.x;
  int64_t stride = blockDim.x;
  int64_t stride1 = gridDim.x;
  while (index < num) {
    int64_t base_read_addr = indptr[index];
    int64_t base_write_addr = indptr_out[index];
    int64_t deg = indptr_out[index + 1] - indptr_out[index];
    for (int64_t i = offset; i < deg; i += stride) {
        ret[base_write_addr + i] = U[base_read_addr + i];
        ret[base_write_addr + i + len] = index;
    }
    index += stride1;
  }
}

 
 
 
 
 
template <typename DType, typename IdType>
__global__ void IndexSelectMultiKernelAligned(
    const DType* const array, const int64_t num_feat, const IdType* const index,
    const int64_t length, const int64_t arr_len, DType* const out,
    const int64_t* perm = nullptr) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < length) {
    int64_t col = threadIdx.x;
    //const int64_t in_row = index[out_row_index];
    const int64_t in_row = index[out_row_index];
    assert(in_row >= 0);
    assert(in_row < arr_len);
    const int64_t idx_offset =
        ((uint64_t)(&array[in_row * num_feat]) % CACHE_LINE_SIZE) /
        sizeof(DType);
    col = col - idx_offset;
    const auto out_row = perm ? perm[out_row_index] : out_row_index;
    while (col < num_feat) {
      if (col >= 0)
        out[out_row * num_feat + col] = array[in_row * num_feat + col];
      col += blockDim.x;
    }
    out_row_index += stride;
  }
}





template <typename DType, typename IdType>
__global__ void IndexSelectKernelAligned(
    const DType* const array, const int64_t num_feat, const IdType* const index,
    const int64_t length, const int64_t arr_len, DType* const out,
    const int64_t* perm = nullptr, const int64_t row_offset = 0) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y + row_offset;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < length) {
    int64_t col = threadIdx.x;
    const int64_t in_row = index[out_row_index];
    //const int64_t in_row = out_row_index;
    assert(in_row >= 0);
    assert(in_row < arr_len);
    const int64_t idx_offset =
        ((uint64_t)(&array[in_row * num_feat]) % CACHE_LINE_SIZE) /
        sizeof(DType);
    col = col - idx_offset;
    const auto out_row = perm ? perm[out_row_index] : out_row_index;
    while (col < num_feat) {
      if (col >= 0)
        __syncthreads();
        out[out_row * num_feat + col] = array[in_row * num_feat + col];
      col += blockDim.x;
    }
    out_row_index += stride;
  }
}



template <typename DType, typename IdType>
__global__ void IndexSelect2D(
    const DType* const array, const int64_t num_feat, const IdType* const index,
    const int64_t length, const int64_t arr_len, DType* const out,
    const int64_t row_size, const int64_t* perm = nullptr) {
  int64_t out_row_index = threadIdx.x + blockDim.x * blockIdx.x;

  const int64_t stride = blockDim.x * gridDim.x;

  while (out_row_index < length) {
    const int64_t in_row = index[out_row_index];
    assert(in_row >= 0 && in_row < arr_len);
    const auto out_row = perm ? perm[out_row_index] : out_row_index;
    //DType* in_row_ptr, out_row_ptr;
    //in_row_ptr = array + in_row;
    //out_row_ptr = out + out_row;
    cudaMemcpy(&out[out_row], &array[in_row], row_size, cudaMemcpyDefault);
    out_row_index += stride;
  }
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CUDA_UVM_ARRAY_INDEX_SELECT_UVM_CUH_
