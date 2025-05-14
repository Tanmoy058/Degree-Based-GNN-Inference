"""Utility functions related to pinned memory tensors."""

from .. import backend as F
from .._ffi.function import _init_api
from ..base import DGLError
import torch

def pin_memory_inplace(tensor):
    """Register the tensor into pinned memory in-place (i.e. without copying).
    Users are required to save the returned dgl.ndarray object to avoid being unpinned.

    Parameters
    ----------
    tensor : Tensor
        The tensor to be pinned.

    Returns
    -------
    dgl.ndarray
        The dgl.ndarray object that holds the pinning status and shares the same
        underlying data with the tensor.
    """
    if F.backend_name in ["mxnet", "tensorflow"]:
        raise DGLError(
            "The {} backend does not support pinning "
            "tensors in-place.".format(F.backend_name)
        )

    # needs to be writable to allow in-place modification
    try:
        nd_array = F.zerocopy_to_dgl_ndarray_for_write(tensor)
        nd_array.pin_memory_()
        return nd_array
    except Exception as e:
        raise DGLError("Failed to pin memory in-place due to: {}".format(e))


def index_select_from_to(src, dst, src_rows, dst_rows):
    _CAPI_IndexSelectInPlace(F.to_dgl_nd(src), F.to_dgl_nd(dst), F.to_dgl_nd(src_rows), F.to_dgl_nd(dst_rows))
  

def Gather(feats, rows, num_threads = torch.tensor([256], dtype = torch.long)):
    """Directly gather rows from a CPU tensor given an indices array on CUDA devices,
    and returns the result on the same CUDA device without copying.

    Parameters
    ----------
    tensor : Tensor
        The tensor.  Must be in pinned memory.
    rows : Tensor
        The rows to gather.  Must be a CUDA tensor.

    Returns
    -------
    Tensor
        The result with the same device as :attr:`rows`.
    """
    row_size = feats.shape[1]
    ret = torch.empty((rows.shape[0], row_size), device = rows.device, dtype = feats.dtype)
    indptr_out = torch.arange(rows.shape[0] + 1, device = rows.device, dtype = torch.long) * row_size
    indptr1 = rows * row_size
    x = F.from_dgl_nd(
        _CAPI_Sample(F.to_dgl_nd(feats), F.to_dgl_nd(indptr1), F.to_dgl_nd(indptr_out), F.to_dgl_nd(ret), F.to_dgl_nd(num_threads))
    )
    #print(x[0], indptr_out[-1], indptr.shape[0], indptr_out.shape[0], rows.shape[0])
    #print(ret)
    return ret





def Sample(U, indptr, rows, num_threads = 256):
    """Directly gather rows from a CPU tensor given an indices array on CUDA devices,
    and returns the result on the same CUDA device without copying.

    Parameters
    ----------
    tensor : Tensor
        The tensor.  Must be in pinned memory.
    rows : Tensor
        The rows to gather.  Must be a CUDA tensor.

    Returns
    -------
    Tensor
        The result with the same device as :attr:`rows`.
    """
    num_threads = torch.tensor([num_threads], dtype = torch.long)
    #[rows, _] = torch.sort(rows)
    indptr1 = indptr[rows]
    indptr2 = indptr[rows+1]
    degs = indptr2 - indptr1
    indptr_out = torch.empty(rows.shape[0] + 1, device = rows.device, dtype = indptr.dtype)
    torch.cumsum(degs, dim = 0, out = indptr_out[1:])
    indptr_out[0] = 0
    ret = torch.empty(indptr_out[-1].cpu().item()*2, device = indptr_out.device, dtype = indptr.dtype)
    x = F.from_dgl_nd(
        _CAPI_Sample(F.to_dgl_nd(U), F.to_dgl_nd(indptr1), F.to_dgl_nd(indptr_out), F.to_dgl_nd(ret), F.to_dgl_nd(num_threads))
    )
    ln = int(indptr_out[-1].cpu().item())
    U_out = ret[0:ln]
    V_out = ret[ln:2*ln]
    V_out = rows[V_out]
    #print(x[0], indptr_out[-1], indptr.shape[0], indptr_out.shape[0], rows.shape[0])
    #print(ret)
    return U_out, V_out, indptr_out
    #return F.from_dgl_nd(
        #_CAPI_DGLGather2D(F.to_dgl_nd(U), F.to_dgl_nd(indptr), F.to_dgl_nd(rows))
    #)


def gather2D(tensor, rows, num_threads = 256):
    """Directly gather rows from a CPU tensor given an indices array on CUDA devices,
    and returns the result on the same CUDA device without copying.

    Parameters
    ----------
    tensor : Tensor
        The tensor.  Must be in pinned memory.
    rows : Tensor
        The rows to gather.  Must be a CUDA tensor.

    Returns
    -------
    Tensor
        The result with the same device as :attr:`rows`.
    """
    num_threads = torch.tensor([num_threads], dtype = torch.long)
    return F.from_dgl_nd(
        _CAPI_DGLGather2D(F.to_dgl_nd(tensor), F.to_dgl_nd(rows), F.to_dgl_nd(num_threads))
    )
    
    
    

def gather_pinned_tensor_rows(tensor, rows):
    """Directly gather rows from a CPU tensor given an indices array on CUDA devices,
    and returns the result on the same CUDA device without copying.

    Parameters
    ----------
    tensor : Tensor
        The tensor.  Must be in pinned memory.
    rows : Tensor
        The rows to gather.  Must be a CUDA tensor.

    Returns
    -------
    Tensor
        The result with the same device as :attr:`rows`.
    """
    return F.from_dgl_nd(
        _CAPI_DGLIndexSelectCPUFromGPU(F.to_dgl_nd(tensor), F.to_dgl_nd(rows))
    )


def scatter_pinned_tensor_rows(dest, rows, source):
    """Directly scatter rows from a GPU tensor given an indices array on CUDA devices,
    to a pinned tensor on the CPU.

    Parameters
    ----------
    dest : Tensor
        The tensor on the CPU to scatter rows to. Must be in pinned memory.
    rows : Tensor
        The rows to scatter. Must be a CUDA tensor with unique entries.
    source : Tensor
        The tensor on the GPU to scatter rows from.
    """
    _CAPI_DGLIndexScatterGPUToCPU(
        F.to_dgl_nd(dest), F.to_dgl_nd(rows), F.to_dgl_nd(source)
    )


_init_api("dgl.ndarray.uvm", __name__)



