# global reference to triplet_loss (will be initialized in .onLoad)
ivis_object <- NULL

.onLoad <- function(libname, pkgname) {

    if ("ivis" %in% reticulate::conda_list()$name) {
        # use superassignment to update global reference to ivis
        reticulate::use_condaenv("ivis")
        
        ivis_object <<- reticulate::import("ivis",
                                        delay_load = TRUE)
    }
}

