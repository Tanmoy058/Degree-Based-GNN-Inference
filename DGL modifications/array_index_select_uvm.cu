/**
 *  Copyright (c) 2019-2022 by Contributors
 * @file array/cuda/uvm/array_index_select_uvm.cu
 * @brief Array index select GPU implementation
 */
#include <dgl/array.h>
#include "../../../runtime/cuda/cuda_common.h"
#include "../array_index_select.cuh"
#include "../utils.h"
#include "./array_index_select_uvm.cuh"


namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <typename DType, typename IdType>
void IndexSelectInPlace(NDArray src, NDArray dst, IdArray src_rows, IdArray dst_rows) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const int64_t num_feat = src->shape[1];
  const int64_t len = src_rows->shape[0];
  DType* src_data = static_cast<DType*>(src->data);
  DType* dst_data = static_cast<DType*>(dst->data);
  const IdType* src_rows_data = static_cast<IdType*>(src_rows->data);
  const IdType* dst_rows_data = static_cast<IdType*>(dst_rows->data);
  
  
  dim3 block(32, 1);
  const dim3 grid((len + block.y - 1) / block.y);
  
  CUDA_KERNEL_CALL(
      IndexSelectInPlaceKernel, grid, block, 0, stream, src_data, dst_data,
      src_rows_data, dst_rows_data, num_feat, len);  
}

// floating point types are treated as their equal width integer types
template void IndexSelectInPlace<int8_t, int64_t>(NDArray, NDArray, IdArray, IdArray);
template void IndexSelectInPlace<int16_t, int64_t>(NDArray, NDArray, IdArray, IdArray);
template void IndexSelectInPlace<int32_t, int64_t>(NDArray, NDArray, IdArray, IdArray);
template void IndexSelectInPlace<int64_t, int64_t>(NDArray, NDArray, IdArray, IdArray);
template void IndexSelectInPlace<int8_t, int32_t>(NDArray, NDArray, IdArray, IdArray);
template void IndexSelectInPlace<int16_t, int32_t>(NDArray, NDArray, IdArray, IdArray);
template void IndexSelectInPlace<int32_t, int32_t>(NDArray, NDArray, IdArray, IdArray);
template void IndexSelectInPlace<int64_t, int32_t>(NDArray, NDArray, IdArray, IdArray);


template <typename DType, typename IdType>
NDArray Sample(NDArray U, IdArray indptr, IdArray indptr_out, IdArray ret, IdArray num_threads) {
    cudaStream_t stream = runtime::getCurrentCUDAStream();
    const int64_t num = indptr_out->shape[0] - 1;
    const int64_t len = (ret->shape[0]) / 2;
    IdType* iod = static_cast<IdType*>(indptr_out->data);
    IdType* id = static_cast<IdType*>(indptr->data);
    DType* U_data = static_cast<DType*>(U->data);
    IdType no_th = static_cast<IdType*>(num_threads->data)[0];
    //const IdType len = iod[num];
    std::vector<int64_t> shape{2};
    NDArray ret2 = NDArray::Empty(shape, indptr->dtype, indptr->ctx);
    IdType* ret_data = static_cast<IdType*>(ret->data);
    const dim3 block(128, 1);
    //const dim3 grid(no_th);
    const dim3 grid((num + block.x) / block.x);
    //ret_data[0] = num;
    CUDA_KERNEL_CALL(
      SampleKernel, grid, block, 0, stream, U_data, ret_data, id, iod, num, len);
    return ret2;
}
// floating point types are treated as their equal width integer types
template NDArray Sample<int8_t, int32_t>(NDArray, IdArray, IdArray, IdArray, IdArray);
template NDArray Sample<int8_t, int64_t>(NDArray, IdArray, IdArray, IdArray, IdArray);
template NDArray Sample<int16_t, int32_t>(NDArray, IdArray, IdArray, IdArray, IdArray);
template NDArray Sample<int16_t, int64_t>(NDArray, IdArray, IdArray, IdArray, IdArray);
template NDArray Sample<int32_t, int32_t>(NDArray, IdArray, IdArray, IdArray, IdArray);
template NDArray Sample<int32_t, int64_t>(NDArray, IdArray, IdArray, IdArray, IdArray);
template NDArray Sample<int64_t, int32_t>(NDArray, IdArray, IdArray, IdArray, IdArray);
template NDArray Sample<int64_t, int64_t>(NDArray, IdArray, IdArray, IdArray, IdArray);


template <typename DType, typename IdType>
NDArray Gather2D(NDArray array, IdArray index, IdArray num_threads) {
  cudaStream_t stream1;
  cudaError_t result;
  result = cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
  //result = cudaStreamCreate(&stream1);
  const int64_t arr_len = array->shape[0];
  const int64_t len = index->shape[0];
  int64_t num_feat = 1;
  std::vector<int64_t> shape{len};
  IdType no_th = static_cast<IdType*>(num_threads->data)[0];

  CHECK(array.IsPinned());
  const DType* array_data = static_cast<DType*>(cuda::GetDevicePointer(array));
  CHECK_EQ(index->ctx.device_type, kDGLCUDA);

  for (int d = 1; d < array->ndim; ++d) {
    num_feat *= array->shape[d];
    shape.emplace_back(array->shape[d]);
  }

  NDArray ret = NDArray::Empty(shape, array->dtype, index->ctx);
  if (len == 0 || arr_len * num_feat == 0) return ret;
  DType* ret_data = static_cast<DType*>(ret->data);

  auto res = Sort(index, cuda::_NumberOfBits(arr_len));
  const IdType* idx_data = static_cast<IdType*>(res.first->data);
  const int64_t* perm_data = static_cast<int64_t*>(res.second->data);

  const dim3 block(32, 1);
  const dim3 grid(no_th);
  CUDA_KERNEL_CALL(
      IndexSelectNA, grid, block, 0, stream1, array_data,
      num_feat, idx_data, len, arr_len, ret_data, perm_data);  

  cudaStreamSynchronize(stream1);
  result = cudaStreamDestroy(stream1);
  return ret;
}

// floating point types are treated as their equal width integer types
template NDArray Gather2D<int8_t, int32_t>(NDArray, IdArray, IdArray);
template NDArray Gather2D<int8_t, int64_t>(NDArray, IdArray, IdArray);
template NDArray Gather2D<int16_t, int32_t>(NDArray, IdArray, IdArray);
template NDArray Gather2D<int16_t, int64_t>(NDArray, IdArray, IdArray);
template NDArray Gather2D<int32_t, int32_t>(NDArray, IdArray, IdArray);
template NDArray Gather2D<int32_t, int64_t>(NDArray, IdArray, IdArray);
template NDArray Gather2D<int64_t, int32_t>(NDArray, IdArray, IdArray);
template NDArray Gather2D<int64_t, int64_t>(NDArray, IdArray, IdArray);

template <typename DType, typename IdType>
NDArray IndexSelectCPUFromGPU(NDArray array, IdArray index) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  //cudaStream_t stream;
  //cudaError_t result;
  //result = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  const int64_t arr_len = array->shape[0];
  const int64_t len = index->shape[0];
  int64_t num_feat = 1;
  std::vector<int64_t> shape{len};

  CHECK(array.IsPinned());
  const DType* array_data = static_cast<DType*>(cuda::GetDevicePointer(array));
  CHECK_EQ(index->ctx.device_type, kDGLCUDA);

  for (int d = 1; d < array->ndim; ++d) {
    num_feat *= array->shape[d];
    shape.emplace_back(array->shape[d]);
  }

  NDArray ret = NDArray::Empty(shape, array->dtype, index->ctx);
  if (len == 0 || arr_len * num_feat == 0) return ret;
  DType* ret_data = static_cast<DType*>(ret->data);

  auto res = Sort(index, cuda::_NumberOfBits(arr_len));
  const IdType* idx_data = static_cast<IdType*>(res.first->data);
  const int64_t* perm_data = static_cast<int64_t*>(res.second->data);
  if (num_feat == 1) {
    const int nt = cuda::FindNumThreads(len);
    const int nb = (len + nt - 1) / nt;
    CUDA_KERNEL_CALL(
        IndexSelectSingleKernel, nb, nt, 0, stream, array_data, idx_data, len,
        arr_len, ret_data, perm_data);
  } else {
    dim3 block(32, 1);
    //while (static_cast<int64_t>(block.x)*2 >= num_feat) {
      //block.x /= 2;
      //block.y *= 2;
    //}
    //const dim3 grid((len + block.y - 1) / block.y);
    const dim3 grid(1024);
    if (num_feat * sizeof(DType) < 2 * CACHE_LINE_SIZE) {
      CUDA_KERNEL_CALL(
          IndexSelectMultiKernel, grid, block, 0, stream, array_data, num_feat,
          idx_data, len, arr_len, ret_data, perm_data);
    } else {
      cudaStreamSynchronize(stream);
      CUDA_KERNEL_CALL(
          IndexSelectMultiKernelAligned, grid, block, 0, stream, array_data,
          num_feat, idx_data, len, arr_len, ret_data, perm_data);
    }
  }
  //cudaStreamSynchronize(stream);
  //result = cudaStreamDestroy(stream);
  return ret;
}

// floating point types are treated as their equal width integer types
template NDArray IndexSelectCPUFromGPU<int8_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<int8_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<int16_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<int16_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<int32_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<int32_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<int64_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<int64_t, int64_t>(NDArray, IdArray);

template <typename DType, typename IdType>
void IndexScatterGPUToCPU(NDArray dest, IdArray index, NDArray source) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const DType* source_data = static_cast<DType*>(source->data);
  const IdType* idx_data = static_cast<IdType*>(index->data);
  const int64_t arr_len = dest->shape[0];
  const int64_t len = index->shape[0];
  int64_t num_feat = 1;
  std::vector<int64_t> shape{len};

  CHECK(dest.IsPinned());
  DType* dest_data = static_cast<DType*>(cuda::GetDevicePointer(dest));
  CHECK_EQ(index->ctx.device_type, kDGLCUDA);
  CHECK_EQ(source->ctx.device_type, kDGLCUDA);

  for (int d = 1; d < source->ndim; ++d) {
    num_feat *= source->shape[d];
  }

  if (len == 0) return;

  if (num_feat == 1) {
    const int nt = cuda::FindNumThreads(len);
    const int nb = (len + nt - 1) / nt;
    CUDA_KERNEL_CALL(
        IndexScatterSingleKernel, nb, nt, 0, stream, source_data, idx_data, len,
        arr_len, dest_data);
  } else {
    dim3 block(256, 1);
    while (static_cast<int64_t>(block.x) >= 2 * num_feat) {
      block.x /= 2;
      block.y *= 2;
    }
    const dim3 grid((len + block.y - 1) / block.y);
    CUDA_KERNEL_CALL(
        IndexScatterMultiKernel, grid, block, 0, stream, source_data, num_feat,
        idx_data, len, arr_len, dest_data);
  }
}

// floating point types are treated as their equal width integer types
template void IndexScatterGPUToCPU<int8_t, int32_t>(NDArray, IdArray, NDArray);
template void IndexScatterGPUToCPU<int8_t, int64_t>(NDArray, IdArray, NDArray);
template void IndexScatterGPUToCPU<int16_t, int32_t>(NDArray, IdArray, NDArray);
template void IndexScatterGPUToCPU<int16_t, int64_t>(NDArray, IdArray, NDArray);
template void IndexScatterGPUToCPU<int32_t, int32_t>(NDArray, IdArray, NDArray);
template void IndexScatterGPUToCPU<int32_t, int64_t>(NDArray, IdArray, NDArray);
template void IndexScatterGPUToCPU<int64_t, int32_t>(NDArray, IdArray, NDArray);
template void IndexScatterGPUToCPU<int64_t, int64_t>(NDArray, IdArray, NDArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
