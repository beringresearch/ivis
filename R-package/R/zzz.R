# global reference to triplet_loss (will be initialized in .onLoad)
ivis_object <- NULL

.onLoad <- function(libname, pkgname) {

    # Overload environment functions due to lack of virtualenv support in Windows
    if (is_windows()){
        env_use <- reticulate::use_condaenv
    } else{
        env_use <- reticulate::use_virtualenv
    }

    # use superassignment to update global reference to ivis
    env_use("ivis")
    
    ivis_object <<- reticulate::import("ivis",
                                    delay_load = TRUE)
}

