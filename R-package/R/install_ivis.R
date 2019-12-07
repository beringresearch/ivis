#' Set up ivis Python package
#' @importFrom reticulate virtualenv_list virtualenv_create virtualenv_remove
#'             conda_list conda_create conda_remove py_install
#' @export

install_ivis <- function(){

    # Overload environment functions due to lack of virtualenv support in Windows
    if (is_windows()){
        env_list <- conda_list
        env_create <- conda_create
        env_remove <- conda_remove
    } else{
        env_list <- virtualenv_list
        env_create <- virtualenv_create
        env_remove <- virtualenv_remove
    }

    cat("Creating a virtual environment (ivis)\n")
    virtual_envs <- env_list()
    if ("ivis" %in% virtual_envs){
      cat("(ivis) environment already exists. The old environment will be updated.")
      env_remove("ivis")
    }

    envname <- "ivis"
    env_create(envname)

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
