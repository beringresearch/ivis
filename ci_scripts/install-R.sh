deactivate
sudo Rscript -e 'install.packages("reticulate", repos="http://cran.us.r-project.org")'
sudo Rscript -e 'install.packages("devtools", repos="http://cran.us.r-project.org")'

python3 -m pip install virtualenv 

sudo Rscript -e 'devtools::install_github("beringresearch/ivis/R-package");library(ivis);install_ivis()'
