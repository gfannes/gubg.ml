# Machine learning functionality

## Optimization

* Make one optimization step process a single gradient
  * This allows different subsets of parameters to be optimized with different algos and instantiations without increasing the gradient computation load
    * Is this possible for SCG?
    * Maybe use async/await for this
* Detect subsets of parameters that have correlated gradients over the past n epochs
  * Maybe some GVM-based approach can gradually and continuously move parameters between subsets
  * Use a dedicated optimization instance for each subset
    * Optmization within each subset can happen independent and thus more efficiently
      * Suppose a function `f(a,b,c,d) = g(a,b)+h(c,d)`. Optimizing `f()` boils down to optimizing `g()` and `h()`
      * If `f()` is to optimized with round-robin line search, first for `a`, then for `b`, ..., the line search for `a` and `b` will not improve `h()` at all. SCG is not a line search per parameter, but it does span all 4 dimensions with its conjugate gradients, some will benefit `g()`, others will benefit `h()`.
      * 2 independent 2-dim SCGs, one for `ab` and one for `cd` will improve both `g()` and `h()` at the same time, while still consuming the same gradient computations.
    * This will only work for optimization algos that do not need the output function itself, but only the gradient.
* All `adagrad`-like optimizations seem to use the elementwise squared gradient, and not the `t-1` with `t`. The latter will take time correlation into account: if a parameter has time-correlated gradient values, it means it is one that can be adjusted more aggressively using a larger learning rate.
