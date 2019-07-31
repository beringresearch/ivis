# R wrapper for the IVIS algorithm

#Installation

R wrapper for `ivis` is provided via the `reticulate` library.
Prior to installation, ensure that `reticulate` is available on your machine.

```R
install.packages("reticulate")
```

The easiest way to install `ivis` is using the `devtools` package:

```R
devtools::install_github("beringresearch/ivis/R-package")
library(ivis)
install_ivis()
```

After ivis is installed into a virtual environment, restart your R session.

Finally, to set environment to tensorflow, add the following line to your environment variables:

```bash
export KERAS_BACKEND=tensorflow
```

## Example

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
