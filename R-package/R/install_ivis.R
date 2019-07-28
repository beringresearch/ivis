#' Set up ivis Python package
#' @importFrom reticulate virtualenv_list virtualenv_create virtualenv_install py_install
#' @export

install_ivis <- function(){    
    cat("Creating a virtual environment (ivis)\n")
    virtual_envs <- virtualenv_list()
    if ("ivis" %in% virtual_envs){
      ("(ivis) environment already exists. Deleting the environment and running install_ivis() again.")
      virtualenv_remove("ivis")
    }

    envname <- "ivis"
    virtualenv_create(envname)

    py_install("ivis", envname="ivis")
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
