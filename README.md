# Rcpp Machine Learning Library

[![](https://cranlogs.r-pkg.org/badges/grand-total/RcppML)](https://cran.r-project.org/package=RcppML)
[![](https://www.r-pkg.org/badges/version-last-release/RcppML)](https://cran.r-project.org/package=RcppML)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

Learn interpretable and generalizable models of big sparse data in R!

|Function|RcppML Class|
|---|---|
Non-negative Matrix Factorization|`RcppML::nmf`
Autoencoding|`RcppML::autoencode`
Neural Networks|`RcppML::neuralnet`
Singular Value Decomposition|`RcppML::svd`
Principal Component Analysis|`RcppML::pca`
Spectral Bipartitioning|`RcppML::bipartition`
Divisive Clustering|`RcppML::dclust`
Non-negative Least Squares|`RcppML::nnls`
Least Squares|`RcppML::solve`
K-means Clustering|`RcppML::kmeans`
K-nearest Neighbors|`RcppML::knn`

All methods are optimized for fast in-core computation on sparse CSC matrices (_i.e._ `Matrix::dgCMatrix`) or dense matrices through an Eigen C++ backend.

**Hyperparameter Tuning:** Most functions support automatic hyperparameter tuning through the `RcppML::tune` class. This class can replace any hyperparameter in the function call and will automatically determine the value that maximizes generalizability based on Wold's Monte-Carlo cross-validation. There are many optimizations built into this tune class, including automatic collapsing with other instances of the tune class in a function call to perform grid-searching, and early termination of unpromising model fits.

Notes to self: Tune constructor contains arguments for `lower`, `upper`, and `values`, `step`, and `min_step`.

**Interpretability:** Non-negativity constraints on hidden outputs are optional, and can be used to increase interpretability of the model. For autoencoders, tied weights can lead to an NMF-like result.  L1 regularization is available for most methods, and increases the sparsity of the weights or hidden outputs, useful for feature engineering.

**Domain-awareness:** Simple graph-convolution is available for most methods, which allows for knowledge in the form of a weighted graph to encourage more similar model weights between samples that are connected in the graph. Knowledge from any domain can be used, whether for integration, meta-modeling, or spatial analysis.

**Masking**: Most methods support masking missing values, whether these are zeros (structural sparsity) or not (provide a separate masking matrix).

**Optimization:** Most methods support the `RcppML::optimizer` class, which implements adam, among other optimizers, to accelerate convergence of algorithms with stochastic updates.

**Minibatch:** Batch updates are an easy way to increase efficiency of parallelization, 

## RcppML::MatrixXd

RcppML uses a special data structure for storing matrices, the `RcppML::Matrix`. This 64-bit matrix reduces memory footprint and increases parallelization efficiency through several behind-the-scenes tricks:  1) storing floating point values with 7-digit precision, 2) dynamically deciding whether to store as sparse or dense, 3) if sparse, storing columns in a jagged array, storing unique values in each column only once, compressing index arrays to smallest byte possible and then bitpacking, and decompressing only one column at a time only when needed, 4) if dense, plugging directly into the Eigen API for fast BLAS.

The R API is fairly minimal, but supports construction from `Matrix::dgCMatrix`, `cbind` with `dgCMatrix`, column-wise slicing, and fully-parallelized reading/writing from/to file.

## Installation

Install the R package from [CRAN](https://cran.r-project.org/web/packages/RcppML/index.html) or the development version from GitHub:

```
install.packages("RcppML")                       # CRAN version
devtools::install_github("zdebruine/RcppML")     # dev version
```

Once installed and loaded, RcppML C++ headers defining classes can be used in C++ files for any R package using `#include <RcppML.hpp>`.

To use the C++ library, clone the repo and `#include <RcppML.hpp>` (which includes `RcppEigen.h`).

## Why NMF
* Useful for dimensional reduction, sparse signature recovery, prediction, transfer learning, dataset integration, and more
* Arguably the simplest possible dimensional reduction because it finds some number of factors that add to reconstruct the input as well as possible. 
* Generates models that are easy to reason about
* Accurately imputes signal dropout (sparsity)
* Prior knowledge can be incorporated as a graph describing feature or sample similarities, or a matrix giving weights for each data point, or a matrix of the same dimensions as `w` or `h` that couples factors to other information
* No need to scale and center, and use all data, not just variable features
* Easy to determine the best rank using cross-validation

## Why not NMF
* Too slow
* Not very robust

RcppML NMF fixes both problems.

## What can it do
* Automatic rank determination for variance-stabilized data
* Fast cross-validation for rank determination
* Masking for input data, `W`, and/or `H`
* Regularize Convex L1 and L2-like (angular/pattern extraction) regularizations to increase sparsity and model stability.
* Specializations for sparse and dense data
* Specializations for symmetric data
* Specialization for rank-2 NMF (faster than rank-2 SVD)
* Can mask zeros (handle as missing)
* Fully parallelized with OpenMP
* Fast stopping criterion based on convergence of the model (cosine similarity of the model across consecutive iterations)
* Diagonal scaling based on the L1 norm of `W` and `H`
* Built-in new xorshift+xoroshiro RNG for transpose-identical matrix generation (useful in masking during cross-validation)

## Quick Start

The `nmf` function is a lightweight wrapper around the full R API:

```
nmf(A, k = NULL, tol = 1e-4, maxit = 100, verbose = 1, L1 = c(0, 0), ...)
```

Note:  the `...` catches some earlier parameters that are no longer implemented, like `diag`. It also catches `seed` (for backwards compatibility with current CRAN version), and if found will set the R seed.

* If `k` is a single integer, a model will be learned at that rank using all data.
* If `k` is an integer vector, models will be learned at each rank and the model with the lowest test set reconstruction error will be returned. The test set is a random speckled pattern of values (6.25% dense) that is randomly with-held during model fitting.
* If `k = NULL`, the rank will be automatically determined using cross-validation.

## Full R API

The R API interfaces with the C++ class and operates in-place by reference.

**Constructor:**
* `nmf_sparse(data)`, constructs a new object of class `nmf`. The `data` matrix is copied and transposed, and pointers to both versions of the matrix are retained in the object. Removing the object from the R environment will invalidate the class.

**Parameter Setters:**
* `$L1(0, 0)` or `$L1(0)` set an L1 penalty in the range `(0, 1]`
* `$L2(0, 0)` or `$L2(0)` set an L2 penalty in the range `(0, 1]`
* `$graph_w(dgCMatrix, numeric)`, `$graph_h(dgCMatrix, numeric)`, a sparse non-negative symmetric [adjacency matrix](https://en.wikipedia.org/wiki/Laplacian_matrix). The graph Laplacian will be computed from this matrix. The second term gives the penalty weight, where a penalty of `1` indicates equal contribution of the NMF and graph objectives to the solution.
* `$mask(dgCMatrix)`, sparse matrix of same dimensions as `data` giving the amount by which each value should be masked during model fitting, where a weight of `1` corresponds to complete masking (handle it as a missing value).
* `$mask_h(matrix)`, `$mask_w(matrix)`, dense matrix of same dimensions as `h` or `w` giving the amount by which each sample or feature should be associated (or "linked") with each factor, usually derived from metadata.
* `$mask_zeros(TRUE)` handle zeros as missing values.
* `$mask_test_set(inverse_density)` entirely mask a random speckled test set consisting of a number of indices corresponding to an inverse density (i.e. inverse density of 16 corresponds to 6.25% density)
* `$seed(integer)` or `$seed(matrix)`, specify a seed to initialize `w` or provide an initialization of `w`. By default, the model uses  `abs(.Random.seed[[3]])` from the R global environment at construction. The seed state is advanced with every random action on the class (i.e. model initialization).
* `$verbose(1)`, set verbosity level (0-3)
* `$threads(0)`, number of threads to use, where `0` corresponds to all detectable threads.

**Fitting and evaluation:**
* `$fit(k, tol = 1e-4, max_iter = 100)` fit the model at a specific rank.
* `$auto_fit(k_init = 10, tol = 1e-4, max_iter = 100)` automatically determine the rank using cross-validation.
* `$error()` calculate mean squared error of the model
* `$test_set_error()` calculate mean squared error of the test set only
* `$train_set_error()` calculate mean squared error of the training set only
* `$predict(newData<dgCMatrix>)`, project the current model onto new data. For square matrices, `w` is used. For rectangular matrices, either `w` or `h` are used depending on dimensions of the new dataset.

It is helpful to use `$verbose(3)` to gain high-level insights into how these functions work.

**Model Getters**
* `$get()` clone the model to a list of `w`, `d`, `h`, `tol`, `iter`, and miscellaneous information including fitting statistics
* `$w()`, `$d()`, `$h()`, `$fit_tol()`, `$fit_iter()` get the `w`, `d`, or `h` components of the model, the fit tolerance, and the number of iterations used to generate the fit.
* `$fit_info()` get cross-validation results and tolerances for fitting iterations, etc.
* `$clone()` clone the model to a new model object

### Code example

Example using R:

```
library(RcppML)
data(hibirds)                 # load a dataset of hawaii bird frequency
                              #   in 10km^2 survey grids
set.seed(123)                 # make random initialization reproducible
m <- nmf_model(hibirds$data)
m$L1(0.01)                    # L1 makes for a little more sparsity
m$auto_fit(k_init = 10)       # automatically find the best rank and fit the model, 
                              #   starting with k = 10
model <- m$get()              # get the final model at the best rank
cv_plot(m)                    # plot the cross-validation results
```

### What happens behind the scenes

The model will initially fit using float precision, but will be automatically upgraded to double when tolerance increases (an indication of lack of numerical stability).

If a random test set is masked, the error will be computed after 5 priming iterations. If the error ever surpasses the error after 5 iterations, the factorization is aborted due to overfitting.

### Vignettes
* The `RcppML::NMF` class: fast and flexible NMF
* The `RcppML::SparseMatrix` class:  zero-copy const reference access to `Matrix` S4 dgCMatrix objects
* The `RcppML::RNG` class:  A two-dimensional linear congruential random number generator using xorshift/xoroshiro128+ for populating transpose-identical dense and sparse matrices all at once or on the fly
* The `RcppML::Cluster` class: fast recursive bipartitioning by rank-2 NMF
