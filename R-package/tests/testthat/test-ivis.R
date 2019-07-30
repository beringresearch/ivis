context("ivis")

test_that("ivis runs on iris dataset", {
            data(iris)
            model <- ivis(k = 3, epochs=1)

            X <- data.matrix(iris[, 1:4])
            X <- scale(X)
            model <- model$fit(X)

            xy <- model$transform(X)
            expect_equal(dim(xy), c(150, 2))
})
