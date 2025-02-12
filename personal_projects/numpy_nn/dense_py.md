# Dense
[github link](https://github.com/colecgulino/numpy-nn/blob/main/nn/dense.py)

The simplest layer we can imagine is a simple Feedforward Linear layer which we call a `Dense` module in the library.

<img src="images/dense.png" width="200">

The signature of the layer initialization is:
```python
def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bias: bool = True,
        name: str = 'Dense'
) -> None:
```

## Forward pass
The parameters of the `Dense` layer are a weight parameter `W` and bias parameter `b`. The weight matrix `W` maps the input of the module to the output space and is of shape `[in_dim, out_dim]`. The bias is a parameter of shape `[out_dim]` which adds some bias values to the output after the weight matrix is applied.

The forward pass of the network is very simple:
$$
y = xW + b
$$

For a simple input $x = [x_1, x_2]$ and parameters:
$$
W = \begin{bmatrix}
w_{11} & w_{12} \\
w_{21} & w_{22}
\end{bmatrix}
$$
$$
b = [b_1, b_2]
$$

The output of the module then:
$$
y.T = \begin{bmatrix}
x_1w_{11} + x_2w_{21} + b_1 \\
x_1w_{12} + x_2w_{22} + b_2
\end{bmatrix}
$$

In Pytorch, we can write this as:
```python
x = x @ self.W + self.b
```

## Backward Pass

In all of our backward passes, we assume that we have a gradient from the upstream level `backwards_gradient` or `din` which is the shape of the output layer of the forward pass. For us this will be the same as shape `y` of `[out_dim]`. 

$$
din = [din_1, din_2]
$$

Given we assume there is a backwards gradient `din` above which contains all the upstream gradients passed down from the chain, rule, we can just calculate the gradient from this layer in isolation.

Using the example from the forward pass, we can calculate how our parameters should change based on the 1. output 2. backwards gradient.

$$
\frac{\partial y}{\partial W}; \frac{\partial y}{\partial b}
$$
