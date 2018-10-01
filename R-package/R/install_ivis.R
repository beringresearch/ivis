#' Set up ivis Python package
#' @importFrom reticulate conda_list conda_create conda_binary conda_install
#' @export

install_ivis <- function(){    
    cat("Creating a conda environment (ivis)\n")
    conda_envs <- conda_list()
    if ("ivis" %in% conda_envs$name){
        stop("(ivis) environment already exists. Delete the environment and run install_ivis() again.")
    }

    envname <- "ivis"
    conda_create(envname)
    condaenv_bin <- function(bin) path.expand(file.path(dirname(conda_binary()), bin))
    packages = c("pip", "tensorflow", "keras", "numpy", "scikit-learn")
    conda_install("ivis", packages, forge = TRUE)

    tmp <- tempdir()
    current_dir <- getwd()
    setwd(tmp)
    system("git clone https://github.com/beringresearch/ivis")
    setwd("ivis")

    cmd <- sprintf("%s%s %s && pip install .",
            ifelse(is_windows(), "", ifelse(is_osx() || is_linux(), "source ",
                "/bin/bash -c \"source ")),
            shQuote(path.expand(condaenv_bin("activate"))),
            envname)
    result <- system(cmd)

  
    setwd(current_dir)
    unlink(tmp)
}   


is_osx <- function(){
    return(Sys.info()["sysname"] == "Darwin")
}

is_linux <- function(){
    return(.Platform$OS.type == "unix")
}

is_windows <- function(){
    return(.Platform$OS.type == "windows")
}