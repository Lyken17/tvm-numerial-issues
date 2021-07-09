import numpy as np

import torch as th
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch import autograd


import tvm
from tvm import auto_scheduler, te
from tvm.contrib.dlpack import to_pytorch_func

from tvm import topi
from tvm.topi.nn import get_pad_tuple
from tvm.topi.nn.pad import pad
from tvm.topi import get_const_int, get_const_tuple, tag
from tvm.topi.utils import simplify


def my_assertion(actual, desired, min_rtol=1, max_rtol=12):
    for i in range(min_rtol, max_rtol):
        try:
            np.testing.assert_allclose(actual, desired, rtol=0.1 ** i)
        except AssertionError as e:
            print(f"Pass atol 1e-{i-1}")
            print(f"Fail atol 1e-{i}")
            return 
    print(f"Pass all atol tests")


if __name__ == "__main__":
    B, M, K, N = 1, 20, 2, 20
    dev = tvm.cpu()

    input = te.placeholder((B, M, K), name="input")
    weight = te.placeholder((B, N, K), name="input")
    out = topi.nn.batch_matmul(input, weight)
    sch = te.create_schedule(out.op)
    tlinear = tvm.build(sch, [input, weight, out])

    idtype = input.dtype
    a = tvm.nd.array(np.random.uniform(size=(B, M, K)).astype(idtype), dev)
    b = tvm.nd.array(np.random.uniform(size=(B, N, K)).astype(idtype), dev)
    c = tvm.nd.array(np.random.uniform(size=(B, M, N)).astype(idtype), dev)
    tlinear(a, b, c)
    
    ta = torch.from_numpy(a.numpy())
    tb = torch.from_numpy(b.numpy())
    tb = tb.transpose(1, 2)
    
    tc = torch.matmul(ta, tb)

    print("Compare TVM with PTH")
    my_assertion(c.numpy(), tc.numpy())

    print("Compare TVM with Numpy")
    na = a.numpy()
    nb = b.numpy()

    nc_slices = [
        np.expand_dims(np.matmul(na[b], nb[b].T), axis=0)
        for b in range(B)
    ]
    nc = np.concatenate(nc_slices, axis=1)
    my_assertion(c.numpy(), nc)

    print("Compare PyTorch with Numpy")
    my_assertion(tc.numpy(), nc)
