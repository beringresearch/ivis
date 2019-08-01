deactivate
sudo Rscript -e 'remotes::install_github("rstudio/reticulate")'
sudo Rscript -e 'install.packages("devtools", repos="http://cran.us.r-project.org")'

sudo apt-get install python3-pip
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv 
python3 -m pip install numpy

sudo Rscript -e 'devtools::install_github("beringresearch/ivis/R-package");library(ivis);install_ivis()'
