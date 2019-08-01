sudo Rscript -e 'install.packages("reticulate")'
sudo Rscript -e 'install.packages("devtools")'
sudo Rscript -e 'devtools::install_github("beringresearch/ivis/R-package");library(ivis);install_ivis()'
