# Rcpp Machine Learning Library

[![](https://cranlogs.r-pkg.org/badges/grand-total/RcppML)](https://cran.r-project.org/package=RcppML)
[![](https://www.r-pkg.org/badges/version-last-release/RcppML)](https://cran.r-project.org/package=RcppML)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

`RcppML` is a C++ library with R bindings for fast **non-negative matrix factorization** and other related methods. RcppML NMF is faster and more flexible than any other NMF implementation.

## Installation

Install the R package from [CRAN](https://cran.r-project.org/web/packages/RcppML/index.html) or the development version from GitHub:

```
install.packages("RcppML")                       # CRAN version
devtools::install_github("zdebruine/RcppML")     # dev version
```

Once installed and loaded, RcppML C++ headers defining classes can be used in C++ files for any R package using `#include <RcppML.hpp>`.

To use the C++ library, clone the repo and `#include <RcppML.hpp>`. You will also need to clone `RcppEigen`.

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

I am in the process of writing vignettes for as many topics as possible, with accompanying publications. Please read and cite accordingly.

## References
[bioRXiv manuscript](https://www.biorxiv.org/content/10.1101/2021.09.01.458620v1) on NMF of single-cell transcriptomics data.

### Code example

Example using R:

```
library(RcppML)
data(hibirds)                 # load a dataset of hawaii bird frequency
                              #   in 10km^2 survey grids
set.seed(123)                 # make random initialization reproducible
m <- nmf(hibirds$data)
m$L1(0.01)                    # L1 makes for a little more sparsity
m$cv() # cross-validate, holding out a 5% dense test set
m$fit(k = 2:20, nstart = 3)   # fit models at all ranks between 2 and 20 for 3 random initializations
model <- m$get_best()         # pick the model at the optimal rank
```

Example using C++:

```
#include <RcppML.hpp>          // also includes Eigen
Eigen::MatrixXd data = Eigen::MatrixXd::Random(1000, 1000);
RcppML::nmf m(data);
m.
```

### Full R API

The R API interfaces with the C++ class and operates in-place by reference.

**Constructor:**
* `nmf(data)` constructs a new object of class `nmf`. The `data` matrix and its transpose is copied to C++ and stored in `float` precision.

**Parameter Setters:**
* `$L1(0, 0)` or `$L1(0)` set an L1 penalty in the range `(0, 1]`
* `$L2(0, 0)` or `$L2(0)` set an L2 penalty in the range `(0, 1]`
* `$graph_w(dgCMatrix, numeric)`, `$graph_h(dgCMatrix, numeric)`, a sparse symmetric [adjacency matrix](https://en.wikipedia.org/wiki/Laplacian_matrix) giving non-negative edge weights. The graph Laplacian will be computed from this matrix. The second term gives the penalty weight, where a penalty of `1` indicates equal contribution of the Euclidean and graph objectives to the solution.
* `$mask(dgCMatrix)`, sparse matrix of same dimensions as `data` giving amount by which each value should be masked during model fitting, where a weight of `1` corresponds to complete masking (handle it as a missing value).
* `$mask_h(matrix)`, `$mask_w(matrix)`, dense matrix of same dimensions as `h` or `w` giving the amount by which each sample or feature should be associated with each factor (e.g. linking), usually derived from some form of metadata.
* `$mask_zeros(TRUE)` handle zeros as missing values.
* `$mask_test_set(inverse_density)` entirely mask a random speckled test set consisting of a number of indices corresponding to an inverse density (i.e. inverse density of 16 corresponds to 6.25% density)
* `$seed(integer)` or `$seed(matrix)`, specify a seed to initialize `w` or provide an initialization of `w`. By default, the model uses  `abs(.Random.seed[[3]])` from the R global environment at construction. The seed state is advanced with every random action on the class (i.e. model initialization).
* `$verbose(1)`, set verbosity level (0-3)
* `$threads(0)`, number of threads to use, where `0` corresponds to all detectable threads.

Any parameter

* `$fit(k, tol = 1e-4, max_iter = 100)`
* `$error()` calculate error of the model
* `$test_set_error()` calculate test set error of the model
* `$predict(newData<dgCMatrix>)`, project the current model onto new data. For square matrices, `w` is used. For rectangular matrices, either `w` or `h` are used depending on dimensions of the new dataset.

Construct a new NMF object using your data as either a dense or sparse matrix, and then set any of the following parameters:

The object only stores a single model, but always stores fit information after each update (tolerance, iteration, seed, and test set error).

```
m <- nmf(data)
```

C++ object stores a vector of fit results.

**Parameters**

**Train**

**Visualize**


### What happens behind the scenes

The model will initially fit using float precision, but will be automatically upgraded to double when tolerance increases (an indication of lack of numerical stability).

If a random test set is masked, the error will be computed after 5 priming iterations. If the error ever surpasses the error after 5 iterations, the factorization is aborted due to overfitting.
