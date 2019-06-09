# R wrapper for the IVIS algorithm

## Installation
R will install ivis into "ivis" conda environment. 

The easiest way to install ivis is using the `devtools` package:

```
devtools::install_github("beringresearch/ivis/R-package")
library(ivis)
install_ivis()
```

After ivis is installed into a conda environment, restart your R session.

Finally, to set environment to tensorflow, add the following line to your environment variables:
```
export KERAS_BACKEND=tensorflow
```

## Example
```
library(ivis)

model <- ivis(k = 3, batch_size = 3)

X = data.matrix(iris[:, 1:4])
model = model$fit(X)

xy = model$transform(X)
```
