from __future__ import absolute_import

import torch as th
import scipy.sparse

# Tensor types
Tensor = th.Tensor
SparseTensor = th.sparse.FloatTensor

# Data types
float16 = th.float16
float32 = th.float32
float64 = th.float64
uint8 = th.uint8
int8 = th.int8
int16 = th.int16
int32 = th.int32
int64 = th.int64

# Operators
tensor = th.tensor
sparse_tensor = th.sparse.FloatTensor
sum = th.sum
max = th.max

def astype(a, ty):
    return a.type(ty)

def asnumpy(a):
    return a.cpu().numpy()

def pack(tensors):
    return th.cat(tensors)

def unpack(x, indices_or_sections=1):
    return th.split(x, indices_or_sections)

def shape(x):
    return x.shape

def isinteger(x):
    return x.dtype in [th.int, th.int8, th.int16, th.int32, th.int64]

unique = th.unique

def gather_row(data, row_index):
    return th.index_select(data, 0, row_index)

def scatter_row(data, row_index, value):
    return data.index_copy(0, row_index, value)

def broadcast_to(x, to_array):
    return x + th.zeros_like(to_array)

def view(x, shape):
    return x.view(shape)

class SPMM(th.autograd.Function):
    @staticmethod
    def forward(ctx, adj_ind, adj_val, feature, matrix_dense_shape):
        ctx.save_for_backward(adj_ind, adj_val, feature)
        ctx.matrix_dense_shape = matrix_dense_shape
        n = feature.shape[0]
        adjmat = th.sparse.FloatTensor(adj_ind, adj_val, matrix_dense_shape)
        return th.spmm(adjmat, feature)

    @staticmethod
    def backward(ctx, grad):
        adj_ind, adj_val, feature = ctx.saved_tensors
        n = feature.shape[0]
        grad_ind = grad_val = grad_feature = None
        if ctx.needs_input_grad[1]:
            a = th.index_select(grad, 0, adj_ind[0])
            b = th.index_select(feature, 0, adj_ind[1])
            grad_val = th.sum(a * b, dim=1).view(-1)
        if ctx.needs_input_grad[2]:
            adjmat = th.sparse.FloatTensor(adj_ind, adj_val, ctx.matrix_dense_shape)
            adjmat = th.transpose(adjmat, 0, 1)
            grad_feature = th.spmm(adjmat, grad)
        return grad_ind, grad_val, grad_feature, None # None for matrix_dense_shape

nonzero = th.nonzero
squeeze = th.squeeze
unsqueeze = th.unsqueeze
reshape = th.reshape
zeros = th.zeros
ones = th.ones
spmm = th.spmm
spmm_grad = SPMM.apply
sort = th.sort
arange = th.arange

def to_context(x, ctx):
    if ctx is None:
        return x
    elif ctx.device == 'gpu':
        th.cuda.set_device(ctx.device_id)
        return x.cuda()
    elif ctx.device == 'cpu':
        return x.cpu()
    else:
        raise RuntimeError('Invalid context', ctx)
