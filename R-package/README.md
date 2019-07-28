# R wrapper for the IVIS algorithm

## Installation
R wrapper for `ivis` is provided via the `reticulate` library. Prior to installation, ensure that `reticulate` is available on your machine.

```R
install.packages("reticulate")
```


The easiest way to install `ivis` is using the `devtools` package:

```
devtools::install_github("beringresearch/ivis/R-package")
library(ivis)
install_ivis()
```

After ivis is installed into a virtual environment, restart your R session.

Finally, to set environment to tensorflow, add the following line to your environment variables:
```
export KERAS_BACKEND=tensorflow
```

## Example
```
library(ivis)

model <- ivis(k = 3)

X = data.matrix(iris[, 1:4])
model = model$fit(X)

xy = model$transform(X)
```
