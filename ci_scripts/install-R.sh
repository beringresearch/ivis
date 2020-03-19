sudo Rscript -e 'remotes::install_github("rstudio/reticulate")'
sudo Rscript -e 'install.packages("covr", repos="http://cran.us.r-project.org")'
sudo Rscript -e 'install.packages("devtools", repos="http://cran.us.r-project.org", dependencies=TRUE)'
sudo Rscript -e 'devtools::install_github("beringresearch/ivis/R-package");library(ivis);install_ivis()'
