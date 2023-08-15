# Tests on the .backward() method of the Tensor object in Tinygrad

Modification is found in ``main.py``.

Tested on CPU:

- Grad output identical between old and new when tested on a weight tensor
- Seems to eke out 1-2% performance when training a small CNN

### View

![view](Screenshot_from_2023-08-15_11-02-11.png)
