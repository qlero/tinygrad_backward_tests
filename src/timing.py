import numpy as np

from time import time
from tinygrad.tensor import Tensor

loss = lambda x, W: x.dot(W).sum().abs()

def time_loop_single_old_backward(inp: np.array,
                                  wgt: np.array,
                                  n_iter: int = 100):
    time_per_backward = []
    for i in range(n_iter):
        x = Tensor(inp, requires_grad = False)
        W = Tensor(wgt, requires_grad = True)
        l = loss(x, W)
        s1 = time()
        l.backward()
        s2 = time()
        time_per_backward.append(s2-s1)
    print(f"\tAvg. elapsed time: {np.mean(time_per_backward):.5f}s",
          f"Std.Dev.: {np.std(time_per_backward):.5f}s",
          sep="; ")

def time_loop_single_new_backward(inp: np.array,
                                  wgt: np.array,
                                  n_iter: int = 100):
    time_per_backward = []
    for i in range(n_iter):
        x = Tensor(inp, requires_grad = False)
        W = Tensor(wgt, requires_grad = True)
        l = loss(x, W)
        s1 = time()
        l.bwd()
        s2 = time()
        time_per_backward.append(s2-s1)
    print(f"\tAvg. elapsed time: {np.mean(time_per_backward):.5f}s",
          f"Std.Dev.: {np.std(time_per_backward):.5f}s",
          sep="; ")

def time_loop_consecutive_old_backward(x: Tensor, 
                                       W: Tensor, 
                                       n_iter: int = 100):
    time_per_backward = []
    start = time()
    for i in range(n_iter):
        l = loss(x, W)
        s1 = time()
        l.backward()
        s2 = time()
        W = W.sub(0.0001 * W.grad) 
        time_per_backward.append(s2-s1)
    end = time()
    print(f"\tTotal elapsed time: {end - start:.2f}s",
          f"Avg. time per backward: {np.mean(time_per_backward):.5f}s",
          f"Std.Dev.: {np.std(time_per_backward):.5f}s",
          sep="; ")
    return W, time_per_backward


def time_loop_consecutive_new_backward(x: Tensor, 
                                       W: Tensor, 
                                       n_iter: int = 100):
    time_per_backward = []
    start = time()
    for i in range(n_iter):
        l = loss(x, W)
        s1 = time()
        l.bwd()
        s2 = time()
        W = W.sub(0.0001 * W.grad) 
        time_per_backward.append(s2-s1)
    end = time()
    print(f"\tTotal elapsed time: {end - start:.2f}s",
          f"Avg. time per backward: {np.mean(time_per_backward):.5f}s",
          f"Std.Dev.: {np.std(time_per_backward):.5f}s",
          sep="; ")
    return W, time_per_backward
