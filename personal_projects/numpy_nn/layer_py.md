# Layer

[github link](https://github.com/colecgulino/numpy-nn/blob/main/nn/layer.py)

This file lays out the interface to all of the other modules
that will be presented in the rest of the neural network layers.

This has two main functions which should be understandable to all of users of popular languages like Pytorch and Jax.

```python
def forward(self, x: np.ndarray) -> tuple[np.ndarray, Cache]:
```
```python
def backward(
    self,
    x: np.ndarray,
    cache: Cache,
    backwards_gradient: np.ndarray,
    gradients: dict[str, np.ndarray]
) -> np.ndarray:
```

These two functions make up the bulk of the functionality of the `Layer` class. The thing to call out is the `Cache` dataclass `Cache = dict[str, np.ndarray]` which is the way we cache out information for the backwards pass.

Another function here is a recurssive function for updating the parameters of a `Layer` and all attributes of that `Layer`. This allows us to pass in specific parameters to update nested `Layer`s.
```python
def update_parameters(self, parameters: dict[str, np.ndarray]) -> None:
```