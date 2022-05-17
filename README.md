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
[bioRXiv manuscript](https://www.biorxiv.org/content/10.1101/2021.09.01.458620v1) on NMF for single-cell experiments.

### Code example

Example using R:

```
library(RcppML)
data(hibirds)                 # load a dataset of hawaii bird frequency
                              #   in 10km^2 survey grids
m <- NMF(hibirds$counts)
m$L1(0.01)                    # L1 makes for a little more sparsity
set.seed(123)                 # make random initialization reproducible
m$fit(k = 2:20)               # fit models at all ranks between 2 and 20
m$
```

Example using C++:

```
#include <RcppML.hpp>          // also includes Eigen
Eigen::MatrixXd data = Eigen::MatrixXd::Random(1000, 1000);
RcppML::nmf m(data);
m.
```



#### R functions
The `nmf` function runs matrix factorization by alternating least squares in the form `A = WDH`. The `project` function updates `w` or `h` given the other, while the `mse` function calculates mean squared error of the factor model.

```{R}
A <- Matrix::rsparsematrix(1000, 100, 0.1) # sparse Matrix::dgCMatrix
model <- RcppML::nmf(A, k = 10, nonneg = TRUE)
h0 <- RcppML::project(A, w = model$w)
RcppML::mse(A, model$w, model$d, model$h)
```

#### C++ class
The `RcppML::MatrixFactorization` class is an object-oriented interface with methods for fitting, projecting, and evaluating linear factor models. It also contains a sparse matrix class equivalent to `Matrix::dgCMatrix` in R.

```{Rcpp}
#include <RcppML.hpp>

//[[Rcpp::export]]
Rcpp::List RunNMF(const Rcpp::S4& A_, int k){
     RcppML::Matrix A(A_); // zero-copy, unlike arma or Eigen equivalents
     RcppML::MatrixFactorization model(k, A.rows(), A.cols());
     model.tol = 1e-5;
     model.fit(A);
     return Rcpp::List::create(
          Rcpp::Named("w") = model.w,
          Rcpp::Named("d") = model.d,
          Rcpp::Named("h") = model.h,
          Rcpp::Named("mse") = model.mse(A));
}
```

## Divisive Clustering
Divisive clustering by rank-2 spectral bipartitioning.
* 2nd SVD vector is linearly related to the difference between factors in rank-2 matrix factorization.
* Rank-2 matrix factorization (optional non-negativity constraints) for spectral bipartitioning **~2x faster** than _irlba_ SVD
* Sensitive distance-based stopping criteria similar to Newman-Girvan modularity, but orders of magnitude faster
* Stopping criteria based on minimum number of samples

#### R functions
The `dclust` function runs divisive clustering by recursive spectral bipartitioning, while the `bipartition` function exposes the rank-2 NMF specialization and returns statistics of the bipartition.

```{R}
A <- Matrix::rsparsematrix(A, 1000, 1000, 0.1) # sparse Matrix::dgcMatrix
clusters <- dclust(A, min_dist = 0.001, min_samples = 5)
cluster0 <- bipartition(A)
```

#### C++ class
The `RcppML::clusterModel` class provides an interface to divisive clustering. In the future, more clustering algorithms may be added.

```{Rcpp}
#include <RcppML.hpp>

//[[Rcpp::export]]
Rcpp::List DivisiveCluster(const Rcpp::S4& A_, int min_samples, double min_dist){
   RcppML::Matrix A(A_);
   RcppML::clusterModel model(A, min_samples, min_dist);
   model.dclust();
   std::vector<RcppML::cluster> clusters = m.getClusters();
   Rcpp::List result(clusters.size());
   for (int i = 0; i < clusters.size(); ++i) {
        result[i] = Rcpp::List::create(
             Rcpp::Named("id") = clusters[i].id,
             Rcpp::Named("samples") = clusters[i].samples,
             Rcpp::Named("center") = clusters[i].center);
   }
   return result;
}
```
