import numpy as np

from time import time
from tinygrad.tensor import Tensor

from src import time_loop_single_old_backward
from src import time_loop_single_new_backward
from src import time_loop_consecutive_old_backward
from src import time_loop_consecutive_new_backward
from src import run_training_mnist

def bwd(self):
    assert self.shape == tuple(), f"backward can only be called for scalar tensors, but it has shape {self.shape})"
    # fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
    # this is "implicit gradient creation"
    self.grad = Tensor(1, device=self.device, requires_grad=False)
    for t0 in reversed(self.deepwalk()):
      assert (t0.grad is not None)
      grads = t0._ctx.backward(t0.grad.lazydata)
      grads = [Tensor(g, device=self.device, requires_grad=False) if g else None
        for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
      for t, g in zip(t0._ctx.parents, grads):
        if g is not None and t.requires_grad:
          assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
          t.grad = g if t.grad is None else (t.grad + g)
      t0._ctx = None

n_iter = 200
weight_size = 4096
random_input = np.random.randn(1, weight_size)
random_weight = np.random.randn(weight_size, 2)

Tensor.bwd = bwd

if __name__ == "__main__":

    x = Tensor(random_input, requires_grad = False)
    W = Tensor(random_weight, requires_grad = True)

    print("Timing Test: old deepwalk")
    time_loop_single_old_backward(random_input, random_weight, n_iter)
    wgt, _  = time_loop_consecutive_old_backward(x, W, n_iter)
    w2 = wgt.numpy()

    del x, W, wgt

    x = Tensor(random_input, requires_grad = False)
    W = Tensor(random_weight, requires_grad = True)

    print("Timing Test: new deepwalk")
    time_loop_single_new_backward(random_input, random_weight, n_iter)
    wgt, _  = time_loop_consecutive_new_backward(x, W, n_iter)
    w1 = wgt.numpy()

    del x, W, wgt
    
    assert np.all(w1 == w2)
    print("Assert weight outputs old v. new identical: TRUE")
    
    print("Timing Test: Training a network with old deepwalk")
    start = time()
    run_training_mnist()
    end = time()
    print(f"\tTraining ran in {end-start:.2f}s")
    
    print("Timing Test: Training a network with new deepwalk")
    start = time()
    run_training_mnist(True)
    end = time()
    print(f"\tTraining ran in {end-start:.2f}s")
