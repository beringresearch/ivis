# R wrapper for the `ivis` algorithm

## Installation

### Prerequisites

R wrapper for `ivis` is provided via the `reticulate` library.
Prior to installation, ensure that `reticulate` is available on your machine.

```R
install.packages("reticulate")
```

Next, install [virtualenv](https://virtualenv.pypa.io/en/latest/) package as it will be used to safely interface with the `ivis` Python package.

Finally, the easiest way to install `ivis` is using the `devtools` package:

### Running install
 
```R
devtools::install_github("beringresearch/ivis/R-package")
library(ivis)
install_ivis()
```

After `ivis` is installed, **restart your R session**. 

Newer versions of Keras use tensorflow as the default backend, however if for some reason this isn't the case, add the following line to your environment variables:

```bash
export KERAS_BACKEND=tensorflow
```

# Example

```R
library(ivis)

model <- ivis(k = 3)

X <- data.matrix(iris[, 1:4])
X <- scale(X)
model <- model$fit(X)

xy <- model$transform(X)
```

Embeddings can now be assessed through a scatterplot:

```R
library(ggplot2)

dat <- data.frame(x=xy[,1],
                  y=xy[,2],
                  species=iris$Species)

ggplot(dat, aes(x=x, y=y)) +
  geom_point(aes(color=species)) +
  theme_classic()
```

## Vignette

The `ivis` package includes a [vignette](https://github.com/beringresearch/ivis/blob/master/R-package/vignettes/ivis_singlecell.Rmd) that demonstrates an example workflow using single-cell RNA-sequencing data.

To compile and install this vignette on your system, you need to first have a working installation of `ivis`.
For this, please follow the instructions [above](#installation).

Once you have a working installation of `ivis`, you can reinstall the package including the compiled vignette using the following command:

```R
devtools::install_github("beringresearch/ivis/R-package", build_vignettes = TRUE, force=TRUE)
```
