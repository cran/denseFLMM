#' Functional Linear Mixed Models for Densely Sampled Data
#'
#' Estimation of functional linear mixed models (FLMMs) for functional data
#' sampled on equal grids based on functional principal component analysis.
#' The implemented models are special cases of the general FLMM
#' \deqn{
#' Y_i(t_d) = \mu(t_d,x_i) + z_i^\top U(t_d) + \epsilon_i(t_d),  i = 1,
#' \ldots,n, d = 1, \ldots, D,}
#' with \eqn{Y_i(t_d)} the value of the response of curve \eqn{i} at observation point
#' \eqn{t_d}, \eqn{\mu(t_d,x_i)} is a mean function, which may depend on covariates
#' \eqn{x_i} and needs to be estimated beforehand. \eqn{z_i} is a covariate
#' vector, which is multiplied with the vector of functional random
#' effects \eqn{U(t_d)}.
#' Usually, the functional random effects will additionally include a smooth error term which
#' is a functional random intercept with a special structure that captures deviations
#' from the mean which are correlated along the support of the functions.
#' In this case, the last block of \eqn{z_i} corresponds to an indicator vector of
#' indicators for each curve and the last block in \eqn{U(t)} consists of curve-specific
#' functional random effects.
#' \eqn{\epsilon_i(t_d)} is independent and identically
#' distributed white noise measurement error with homoscedastic,
#' constant variance.\cr\cr
#' The vector-valued functional random effects can be subdivided into \eqn{H}
#' independent blocks of functional random effects \deqn{U(t_d) = (U_1(t_d)^\top, \ldots,
#' U_H(t_d)^\top)^\top,} with \eqn{U_g(t_d)} and \eqn{U_h(t_d)} independent
#' for \eqn{g \neq h}. Each block \eqn{U_h(t_d)} further contains \eqn{L^{U_h}} independent
#' copies \eqn{U_{gl}(t_d)}, \eqn{l=1, \ldots, L^{U_h}}, of a vector-valued stochastic process with
#' \eqn{\rho^{U_h}} vector components \eqn{U_{gls}(t_d)}, \eqn{s = 1,\ldots, \rho^{U_h}}.
#' The total number of functional random effects then amounts to \eqn{q = \sum_{h=1}^H L^{U_h}\rho^{U_h}}.
#' \cr\cr
#' The code implements a very general functional linear mixed model for \eqn{n}
#' curves observed at \eqn{D} grid points. Grid points are assumed to be
#' equidistant and so far no missings are assumed.
#' The curves are assumed to be centered. The functional random effects for each grouping
#' factor are assumed to be correlated (e.g., random intercept and
#' slope curves). The code can handle group-specific functional random
#' effects including group-specific smooth errors. Covariates are assumed to be standardized. \cr
#'
#' @param Y \eqn{n x D} matrix of \eqn{n} curves observed on \eqn{D} grid points.
#' \code{Y} is assumed to be centered by its mean function.
#' @param gridpoints vector of grid points along curves, corresponding to columns of \code{Y}.
#' Defaults to \code{matrix(1, nrow(Y), 1)}.
#' @param Zlist list of length \eqn{H} of \eqn{\rho^{U_g}} design matrices
#' \eqn{Z_{\cdot s^{U_g}}},
#' \eqn{g = 1,\ldots, H}, \eqn{s = 1,\ldots, \rho^{U_g}}. Needed instead of \code{Zvars} and \code{groups}
#' if group-specific functional random effects are present. Defaults to \code{NA},
#' then \code{Zvars} and \code{groups} needed.
#' @param G number of grouping factors not used for estimation of error variance.
#' Needed if \code{Zlist} is used instead of \code{Zvars} and \code{groups}. Defaults to \code{NA}.
#' @param Lvec vector of length \eqn{H} containing the number of levels for each grouping factor.
#' Needed if \code{Zlist} is used instead of \code{Zvars} and \code{groups}. Defaults to \code{NA}.
#' @param groups \eqn{n \times G} matrix with \eqn{G} grouping factors for the rows of \code{Y},
#' where \eqn{G} denotes the number of random grouping factors not used for the estimation
#' of the error variance. Defaults to \code{groups = matrix(1, nrow(Y), 1)}. Set to \code{NA} when
#' \code{Zlist} is used to specify group-specific functional random effects.
#' @param Zvars list of length \eqn{G} with \eqn{n \times \rho^{U_g}} matrices of random variables
#' for grouping factor \eqn{g}, where \eqn{G} denotes the number of random grouping factors not
#' used for the estimation of the error variance. Random curves for each grouping
#' factor are assumed to be correlated (e.g., random intercept and slope).
#' Set to \code{NA} when \code{Zlist} is used to specify group-specific functional random effects.
#' @param L pre-specified level of variance explained (between 0 and 1), determines
#' number of functional principal components.
#' @param NPC vector of length \eqn{H} with number of functional principal components to
#' keep for each functional random effect. Overrides \code{L} if not \code{NA}. Defaults to \code{NA}.
#' @param smooth \code{TRUE} to add smoothing of the covariance matrices, otherwise
#' covariance matrices are estimated using least-squares. Defaults to \code{FALSE}.
#' @param bf number of marginal basis functions used for all smooths.
#' Defaults to \code{bf = 10}.
#' @param smoothalg smoothing algorithm used for covariance smoothing.
#' Available options are \code{"gamm"}, \code{"gamGCV"}, \code{"gamREML"}, \code{"bamGCV"},
#' \code{"bamREML"}, and \code{"bamfREML"}. \code{"gamm"} uses REML estimation based on function \code{\link{gamm}} in
#' \code{R}-package \code{\link{mgcv}}. \code{"gamGCV"} and \code{"gamREML"} use GCV and REML
#' estimation based on function \code{\link{gam}} in \code{R}-package \code{\link{mgcv}}, respectively.
#' \code{"bamGCV"}, \code{"bamREML"}, and \code{"bamfREML"} use GCV, REML, and a fast REML estimation
#' based on function \code{\link{bam}} in \code{R}-package \code{\link{mgcv}}, respectively.
#' Defaults to \code{"gamm"}.
#'
#' @details The model fit for centered curves \eqn{Y_i(.)} is \deqn{Y = ZU + \epsilon,}
#' with \eqn{Y = [Y_i(t_d)]_{i = 1, \ldots, n, d = 1, \ldots, D}},
#' \eqn{Z} consisting of
#' \eqn{H} blocks \eqn{Z^{U_h}} for \eqn{H} grouping factors, \eqn{Z = [Z^{U_1}|\ldots| Z^{U_H}]},
#' and each \eqn{Z^{U_h} = [Z_1^{U_h} |\ldots| Z_{\rho^{U_h}}^{U_h}]}. \eqn{U} is row-wise divided
#' into blocks \eqn{U_1,\ldots, U_H}, corresponding to \eqn{Z}.\cr
#' In case no group-specific functional random effects are specified, column \eqn{j} in \eqn{Z_{s}^{U_g}}, \eqn{s=1,\ldots,\rho^{U_g}},
#' is assumed to be equal to the \eqn{s}th variable (column) in \code{Zvars[[g]]} times an
#' indicator for the \eqn{j}th level of grouping factor \eqn{g}, \eqn{g=1,\ldots,G}. \cr
#' Note that \eqn{G} here denotes the number of random grouping factors not used for the estimation
#' of the error variance, i.e., all except the smooth error term(s).
#' The total number of grouping factors is denoted by \eqn{H}. \cr\cr
#' The estimated eigenvectors and eigenvalues are rescaled to ensure that the approximated eigenfunctions are
#' orthonormal with respect tot the \eqn{L^2}-inner product.\cr\cr
#' The estimation of the error variance takes place in two steps. In case of smoothing (\code{smooth = TRUE}),
#' the error variance is first estimated as the average difference of the raw and the smoothed diagonal values.
#' In case no smoothing is applied, the estimated error variance is zero.
#' Subsequent to the eigen decomposition and selection of the eigenfunctions to keep for each grouping factor,
#' the estimated error variance is recalculated in order to capture the left out variability due to the truncation
#' of the infinite Karhunen-Loeve expansions.
#'
#' @return The function returns a list containing the input arguments
#' \code{Y}, \code{gridpoints}, \code{groups}, \code{Zvars}, \code{L}, \code{smooth}, \code{bf},
#' and \code{smoothalg}. It additionally contains:
#' \itemize{
#' \item \code{Zlist}  either the input argument \code{Zlist} or if set to \code{NA},
#' the computed list of list of design matrices \eqn{Z_{\cdot s}^{U_g}},
#' \eqn{g=1,\ldots,H}, \eqn{s = 1,\ldots, \rho^{U_g}}.
#' \item \code{G}   either the input argument \code{G} or if set to \code{NA},
#' the computed number of random grouping factors \eqn{G} not used for the estimation of the error variance.
#' \item \code{Lvec}  either the input argument \code{Lvec} or if set to \code{NA},
#' the computed vector of length \eqn{H} containing the number of levels
#' for each grouping factor (including the smooth error(s)).
#' \item \code{NPC}  either the input argument \code{NPC} or if set to \code{NA},
#' the number of functional principal components kept for each functional random effect (including the smooth error(s)).
#' \item \code{rhovec} vector of length \eqn{H} of number of random effects for each grouping factor (including the smooth error(s)).
#' \item \code{phi} list of length \eqn{H} of \eqn{D x N^{U_g}} matrices containing
#' the \eqn{N^{U_g}} functional principal components kept per grouping factor (including the smooth error(s)).
#' \item \code{sigma2} estimated measurement error variance \eqn{\sigma^2}.
#' \item \code{nu} list of length \eqn{H} of \eqn{N^{U_g} x 1} vectors of estimated eigenvalues
#' \eqn{\nu_k^{U_g}}.
#' \item \code{xi} list of length \eqn{H} of \eqn{L^{U_g} x N^{U_g}} matrices containing
#' the predicted random basis weights. Within matrices, basis weights are ordered corresponding
#' to the ordered levels of the grouping factors, e.g., a grouping factor with levels 4, 2, 3, 1
#' (\eqn{L^{U_g} = 4}) will result in rows in \code{xi[[g]]} corresponding to levels 1, 2, 3, 4.
#' \item \code{totvar} total average variance of the curves.
#' \item \code{exvar} level of variance explained by the selected functional principal components (+ error variance).
#' }
#' @author Sonja Greven, Jona Cederbaum
#'
#' @keywords models, FPCA
#'
#' @examples
#' # fit model with group-specific functional random intercepts for two groups
#' # and a non group-specific smooth error, i.e., G = 2, H = 1.
#'
#' ################
#' # load libraries
#' ################
#' require(mvtnorm)
#' require(Matrix)
#' set.seed(123)
#'
#' #########################
#' # specify data generation
#' #########################
#' nus <- list(c(0.5, 0.3), c(1, 0.4), c(2)) # eigenvalues
#' sigmasq <- 2.5e-05 # error variance
#' NPCs <- c(rep(2, 2), 1) # number of eigenfunctions
#' Lvec <- c(rep(2, 2), 480) # number of levels
#' H <- 3 # number of functional random effects (in total)
#' G <- 2 # number of functional random effects not used for the estimation of the error variance
#' gridpoints <- seq(from = 0, to = 1, length = 100) # grid points
#' class_nr <- 2 # number of groups
#'
#' # define eigenfunctions
#' #######################
#' funB1<-function(k,t){
#'   if(k == 1)
#'     out <- sqrt(2) * sin(2 * pi * t)
#'   if(k == 2)
#'     out <- sqrt(2) * cos(2 * pi * t)
#'   out
#' }
#'
#' funB2<-function(k,t){
#'   if(k == 1)
#'     out <- sqrt(7) * (20 * t^3 - 30 * t^2 + 12 * t - 1)
#'   if(k == 2)
#'     out <- sqrt(3) * (2 * t - 1)
#'   out
#' }
#'
#' funE<-function(k,t){
#'   if(k == 1)
#'     out <- 1 + t - t
#'   if(k == 2)
#'     out <- sqrt(5) * (6 * t^2 - 6 * t + 1)
#'   out
#' }
#'
#' ###############
#' # generate data
#' ###############
#' D <- length(gridpoints) # number of grid points
#' n <- Lvec[3] # number of curves in total
#'
#' class <- rep(1:class_nr, each = n / class_nr)
#' group1 <- rep(rep(1:Lvec[1], each = n / (Lvec[1] * class_nr)), class_nr)
#' group2 <- 1:n
#'
#' data <- data.frame(class = class, group1 = group1, group2 = group2)
#'
#' # get eigenfunction evaluations
#' ###############################
#' phis <- list(sapply(1:NPCs[1], FUN = funB1, t = gridpoints),
#'              sapply(1:NPCs[2], FUN = funB2, t = gridpoints),
#'              sapply(1:NPCs[3], FUN = funE, t = gridpoints))
#'
#' # draw basis weights
#' ####################
#' xis <- vector("list", H)
#' for(i in 1:H){
#' if(NPCs[i] > 0){
#'  xis[[i]] <- rmvnorm(Lvec[i], mean = rep(0, NPCs[i]), sigma = diag(NPCs[i]) * nus[[i]])
#'  }
#' }
#'
#' # construct functional random effects
#' #####################################
#' B1 <- xis[[1]] %*% t(phis[[1]])
#' B2 <- xis[[2]] %*% t(phis[[2]])
#' E <- xis[[3]] %*% t(phis[[3]])
#'
#' B1_mat <- B2_mat <- E_mat <- matrix(0, nrow = n, ncol = D)
#' B1_mat[group1 == 1 & class == 1, ] <- t(replicate(n =  n / (Lvec[1] * class_nr),
#' B1[1, ], simplify = "matrix"))
#' B1_mat[group1 == 2 & class == 1, ] <- t(replicate(n =  n / (Lvec[1] * class_nr),
#' B1[2, ], simplify = "matrix"))
#' B2_mat[group1 == 1 & class == 2, ] <- t(replicate(n =  n / (Lvec[1] * class_nr),
#' B2[1, ], simplify = "matrix"))
#' B2_mat[group1 == 2 & class == 2, ] <- t(replicate(n =  n / (Lvec[1] * class_nr),
#' B2[2, ], simplify = "matrix"))
#' E_mat <- E
#'
#' # draw white noise measurement error
#' ####################################
#' eps <- matrix(rnorm(n * D, mean = 0, sd = sqrt(sigmasq)), nrow = n, ncol = D)
#'
#' # construct curves
#' ##################
#' Y <- B1_mat + B2_mat + E_mat + eps
#'
#' #################
#' # construct Zlist
#' #################
#' Zlist <- list()
#' Zlist[[1]] <- Zlist[[2]] <- Zlist[[3]] <- list()
#'
#' d1 <- data.frame(a = as.factor(data$group1[data$class == 1]))
#' Zlist[[1]][[1]] <- rbind(sparse.model.matrix(~ -1 + a, d1),
#'   matrix(0, nrow = (1 / class_nr * n), ncol = (Lvec[1])))
#'
#' d2 <- data.frame(a = as.factor(data$group1[data$class == 2]))
#' Zlist[[2]][[1]] <- rbind(matrix(0, nrow = (1 / class_nr * n),
#'   ncol = (Lvec[2])), sparse.model.matrix(~ -1 + a, d2))
#'
#' d3 <- data.frame(a = as.factor(data$group2))
#' Zlist[[3]][[1]] <- sparse.model.matrix(~ -1 + a, d3)
#'
#' ################
#' # run estimation
#' ################
#' results <- denseFLMM(Y = Y, gridpoints = gridpoints, Zlist = Zlist, G = G, Lvec = Lvec,
#'                   groups = NA, Zvars = NA, L = 0.99999, NPC = NA,
#'                   smooth = FALSE)
#'
#' ###############################
#' # plot estimated eigenfunctions
#' ###############################
#' with(results, matplot(gridpoints, phi[[1]], type = "l"))
#' with(results, matplot(gridpoints, phi[[2]], type = "l"))
#' with(results, matplot(gridpoints, phi[[3]], type = "l"))
#'
#' @export
#' @import methods parallel mgcv MASS Matrix
#' @importFrom grDevices terrain.colors
#' @importFrom stats approx as.formula coef coefficients fitted predict reshape var weights
#' @importFrom utils packageVersion
#' @seealso For the estimation of functional linear mixed models for irregularly
#' or sparsely sampled data based on functional principal component analysis,
#' see function \code{sparseFLMM} in package \code{sparseFLMM}.
#'
denseFLMM <- function(Y, gridpoints = 1:ncol(Y), Zlist = NA, G = NA,
                      Lvec = NA, groups = matrix(1, nrow(Y), 1), Zvars, L = NA, NPC = NA,
                      smooth = FALSE, bf = 10, smoothalg = "gamm"){

  ###########################################################################
  # checks for consistency of input
  ###########################################################################
  if(all(is.na(Zlist)) & all(is.na(groups))){
    stop("either Zlist or groups and Zvars must be specified")
  }
  if(all(is.na(Zlist)) & all(is.na(Zvars))){
    stop("either Zlist or groups and Zvars must be specified")
  }
  if(all(is.na(Zlist))){
    if(nrow(Y) != nrow(groups)){
      stop("The number of rows in Y needs to agree with the number
           of rows in the grouping matrix")
    }
    if(ncol(groups) != length(Zvars)){
      stop("the number of grouping factors has to correspond to
           the number of groups of random variables")
    }
    if(!prod(sapply(seq(len = length(Zvars)),
                    function(r) nrow(Zvars[[r]])) == nrow(Y))){
      stop("the number of rows in Y needs to agree with the number
           of rows in the matrices of random variables")
    }
    if(prod(!is.na(NPC))){
      if(!prod(sapply(seq(len = length(NPC)),
                      function(N){!(NPC[N] > floor(NPC[N]))}))){
        warning("NPC contains not only integers, will use rounded values")
        NPC <- round(NPC)
      }
      if(length(NPC) != ncol(groups) + 1){
        warning("the length of NPC has to correspond to the
                number of groups + 1, will repeatedly use last value")
        temp <- length(NPC)
        NPC <- c(NPC, rep(NPC[temp], ncol(groups) + 1 - temp))
      }
    }
  }else{
    # each matrix in Zlist should be of dim nrow(Y) x L^U_g
    check_dims <- function(Zlist, Y, Lvec){
      zdim <- list()
      for(z in seq_along(Zlist)){
        zdim[[z]] <- do.call(rbind, (lapply(Zlist[[z]], dim)))
      }
      zdim_un <- do.call(rbind, zdim)
      if(isTRUE(all.equal(zdim_un[, 1], rep(zdim_un[1, 1], nrow(zdim_un))))){
        x <- isTRUE(all.equal(zdim_un[1, 1], nrow(Y)))
        if(!x)
          stop("the number of rows of each matrix in Zlist need to
                 correspond to the number of rows of Y")
      }else{
        stop("the number of rows of each matrix in Zlist need to
               correspond to the number of rows of Y")
      }
      y <- (all(Lvec == zdim_un[, 2]))
      if(!y)
        stop("the number of columns of each matrix in Zlist need to
               correspond to the respective number in Lvec")
    }
    check_dims(Zlist = Zlist, Y = Y, Lvec = Lvec)
  }
  if(ncol(Y) != length(gridpoints)){
    stop("the number of columns in Y needs to agree with the length of
         the gridpoints vector")
  }
  if(is.na(L) & (!prod(!is.na(NPC)))){
    warning("as both L and part of NPC are missing, will default
            to L = 0.9, NPC = NA")
    L <- 0.9
    NPC <- NA
  }
  if(!is.na(L) & (prod(!is.na(NPC)))){
    warning("NPC will override choice of L")
  }
  if(!is.na(L)){
    if(L > 1 | L < 0){
      stop("the level of explained variance needs to be between 0 and 1")
    }
  }
  if(smooth)
    stopifnot(smoothalg %in% c("gamm", "gamGCV", "gamREML", "bamGCV",
                               "bamREML", "bamfREML"))

  ###########################################################################
  # set up
  ###########################################################################
  message("set up")
  D <- ncol(Y)   # number of points per curve
  n <- nrow(Y)   # overall number of curves

  if(all(is.na(Zlist))){  # no group-specific functional random effects
    G <- length(Zvars)    # number of grouping factors
    rhovec <- 1
    Lvec <- n
    if(G > 0){
      # number of random effects for grouping factor 1,..., G
      rhovec <- c(sapply(1:G, function(g){ncol(Zvars[[g]])}), 1)

      # number of levels of grouping factors 1,..., G
      Lvec <- c(sapply(1:G, function(g){nlevels(as.factor(groups[, g]))}), n)
    }
  }else{
    H <- length(Zlist)
    rhovec <- unlist(lapply(Zlist, length))
  }
  sq2 <- sum(rhovec^2)

  # construct H, Zlist when no group-specific
  # functional random effects are present
  ###########################################
  if(all(is.na(Zlist))){  # define H as G + 1 (for smooth error)
    H <- G + 1
  }
  if(all(is.na(Zlist))){
    message("construct Zlist")
    foroneq <- function(g, q){
      gp1 <- as.factor(groups[, g])
      sparse.model.matrix(~ gp1 * Zvars[[g]][, q] - gp1 - Zvars[[g]][, q] - 1)
    }
    Zlist <- lapply(seq(len = G), function(g){lapply(1:rhovec[g],
                                                     function(q) foroneq(g, q))})
    Zlist[[H]] <- list(Diagonal(n))
  }

  # assume a centered Y for now, no estimation of mean function
  Y.tilde <- Y

  ###########################################################################
  # estimation
  ###########################################################################

  # estimate covariance functions using least squares
  ###################################################
  message("estimate covariance(s)")
  gcyc <- rep(1:(H), rhovec^2)
  qcyc <- unlist(sapply(rhovec, FUN = function(q){rep(1:q, each = q)}))
  pcyc <- unlist(sapply(rhovec, FUN = function(p){rep(1:p, p)}))
  XtXentry <- function(ro, co){
    A1 <-
      if(H == G + 1){ # if no group-specific smooth errors
        # avoid unnecessary multiplications with pseudo-Design
        # Zlist[[H]] (= Identity) for residuals:
        if(gcyc[co] == H){
          Zlist[[gcyc[ro]]][[pcyc[ro]]]
        }else{
          if(gcyc[ro] == H){
            t(Zlist[[gcyc[co]]][[pcyc[co]]])
          }else{
            crossprod(Zlist[[gcyc[co]]][[pcyc[co]]],
                      Zlist[[gcyc[ro]]][[pcyc[ro]]])
          }
        }
      }else{
        crossprod(Zlist[[gcyc[co]]][[pcyc[co]]],
                  Zlist[[gcyc[ro]]][[pcyc[ro]]])
      }
    A2 <-
      if(H == G + 1){ # if no group-specific smooth errors
        # avoid unnecessary multiplications with pseudo-Design
        # Zlist[[H]] (= Identity) for residuals:
        if(gcyc[co] == H){
          Zlist[[gcyc[ro]]][[qcyc[ro]]]
        }else{
          if(gcyc[ro] == H){
            t(Zlist[[gcyc[co]]][[qcyc[co]]])
          }else{
            crossprod(Zlist[[gcyc[co]]][[qcyc[co]]],
                      Zlist[[gcyc[ro]]][[qcyc[ro]]])
          }
        }
      }else{
        crossprod(Zlist[[gcyc[co]]][[qcyc[co]]],
                  Zlist[[gcyc[ro]]][[qcyc[ro]]])
      }

    # get trace tr(A1'A2) without computing off-diagonal elements in A1'A2
    traceA1tA2 <- function(A1, A2){
      ret <- if(all(dim(A1) == dim(A2))){
        sum(rowSums(A1 * A2))
      }else{
        # use tr(A1'A2) = tr(A2'A1) to shorten loop
        if(ncol(A1) < nrow(A1)){
          sum(sapply(1:ncol(A1),
                     function(i) as.numeric(crossprod(A1[, i, drop = F],
                                                      A2[, i, drop = F])), simplify = T))
        }else{
          sum(sapply(1:nrow(A1),
                     function(i) as.numeric(tcrossprod(A1[i, , drop = F],
                                                       A2[i,, drop = F])), simplify = T))
        }
      }
      return(ret)
    }
    return(traceA1tA2(A1, A2))
  }

  # define useful function similar to outer
  # does not need function to work for vectors
  # or to return scalars
  ############################################
  matrixouter <- function(rows, cols, FUN){
    FUN <- match.fun(FUN)
    # use rbind, cbind to preserve sparsity patterns
    do.call(rbind, lapply(rows, function(g)
      do.call(cbind, lapply(cols, function(h) FUN(g, h)))))
  }

  XtX <- matrixouter(seq(len = sq2), seq(len = sq2), FUN = "XtXentry")

  Xtcentry <- function(ro){
    if(gcyc[ro] == H & H == (G + 1)){
      return(as.vector(crossprod(Y.tilde)))
    }else{
      A1 <- crossprod(Zlist[[gcyc[ro]]][[pcyc[ro]]], Y.tilde)
      A2 <- crossprod(Zlist[[gcyc[ro]]][[qcyc[ro]]], Y.tilde)
      return(as.vector(crossprod(A1, A2)))
    }
  }

  Xtc <- do.call(rbind, lapply(seq(len = sq2), function(h){Xtcentry(h)}))
  Ktilde <- solve(XtX, Xtc)

  # smoothing of covariances
  ##########################
  # set up row and column variables for bivariate smoothing
  rowvec <- rep(gridpoints, each = D)
  colvec <- rep(gridpoints, D)

  cum_rhovec2 <- cumsum(rhovec^2)

  if(H == G + 1){ # if no group-specific smooth errors
    # diagonal of smooth error and sigma^2
    diago <- diag(matrix(Ktilde[sq2, ], D, D))
  }else{
    diagos <- list()
    use <- cum_rhovec2[(G + 1):H]
    for(k in seq(along = use)){
      # diagonal of all smooth error terms and sigma^2
      diagos[[k]] <- diag(matrix(Ktilde[use[k], ], D, D))
    }
  }

  if(smooth){
    message("smooth covariance(s)")
    if(H == G + 1){ # if no group-specific smooth errors
      # if smoothing is selected (method described in the paper)
      # do not use diagonal of smooth error + sigma^2
      # in smoothing the smooth error
      Ktilde[sq2, as.logical(diag(D))] <- rep(NA, D)
    }else{
      use <- cum_rhovec2[(G + 1):H]
      for(k in seq(along = use)){
        # do not use diagonals of smooth errors + sigma^2
        # in smoothing of the smooth errors
        Ktilde[use[k], as.logical(diag(D))] <- rep(NA, D)
      }
    }

    Km <- t(sapply(1:sq2, function(r){
      m <- switch(smoothalg,
                  "gamm" = gamm(Ktilde[r, ] ~ te(rowvec, colvec, k = bf))$gam,,
                  "gamREML" = gam(Ktilde[r, ] ~ te(rowvec, colvec, k = bf),
                                  method = "REML"),
                  "gamGCV" = gam(Ktilde[r, ] ~ te(rowvec, colvec, k = bf),
                                 method = "GCV.Cp"),
                  "bamGCV" = bam(Ktilde[r, ] ~ te(rowvec, colvec, k = bf),
                                 method = "GCV.Cp"),
                  "bamREML" = bam(Ktilde[r, ] ~ te(rowvec, colvec, k = bf),
                                  method = "REML"),
                  "bamfREML" = bam(Ktilde[r, ] ~ te(rowvec, colvec, k = bf),
                                   method = "fREML"))

      return(predict(m, newdata = data.frame(rowvec = rowvec,
                                             colvec = colvec)))
    }))
  }else{
    Km <- Ktilde
  }

  # save covariance functions in matrix form, symmetrize
  ######################################################
  onecov <- function(g){
    covcomp <- function(s, r){
      matrix(Km[sum(rhovec[seq(len = g - 1)]^2) + (s - 1) * rhovec[g] + r, ],
             D, byrow = TRUE)
    }
    matrixouter(1:rhovec[g], 1:rhovec[g], FUN = "covcomp")
  }
  K <- lapply(1:(H), "onecov")

  # symmetrize covariance matrices
  K <- lapply(1:(H), function(g){(K[[g]] + t(K[[g]])) / 2})

  # estimation of error variance
  ##############################
  message("estimate error variance")
  if(H == (G + 1)){ # if no group-specific errors
    sigma2.hat <- max(mean((diago -
                              diag(K[[H]]))[floor(D * 0.2):ceiling(D * 0.8)],
                           na.rm = TRUE), 0)  # avoid boundary issues
  }else{
    sigma2.hat_parts <- numeric()
    use <- (G + 1):H
    for(k in seq(along = use)){
      sigma2.hat_parts[k] <- max(mean((diagos[[k]] -
                                         diag(K[[use[k]]]))[floor(D * 0.2):ceiling(D * 0.8)],
                                      na.rm = TRUE), 0) # avoid boundary issues
    }
    sigma2.hat <- max(mean(sigma2.hat_parts, na.rm = TRUE), 0)
  }

  # get sigmas2.hat times length of grid
  sigma2.hat_int <- (max(gridpoints) - min(gridpoints)) * sigma2.hat

  # estimate eigenfunctions and eigenvalues
  #########################################
  # get interval width of equidistant grid
  interv <- gridpoints[2] - gridpoints[1]

  message("make eigen decomposition(s)")
  eigen.list <- lapply(1:(H), function(g) eigen(K[[g]], symmetric = TRUE))
  nu.hat <- lapply(1:(H), function(g) eigen.list[[g]]$values * interv)

  # compute variance explained
  ############################
  total.variance <- sigma2.hat_int + sum(unlist(nu.hat) * (unlist(nu.hat) > 0))

  # choose number of functional principal components
  ##################################################
  if(any(is.na(NPC))){
    message("get truncation level(s)")
    explained.variance <- sigma2.hat_int / total.variance
    NPC <- rep(0, H)   # add components with decreasing variance
    while(explained.variance < L){
      maxl <- sapply(1:(H), function(g){nu.hat[[g]][NPC[[g]] + 1]})
      maxg <- match(max(maxl), maxl)
      NPC[[maxg]] <- NPC[[maxg]] + 1
      explained.variance <- explained.variance +
        nu.hat[[maxg]][NPC[[maxg]]] / total.variance
    }
  }

  phi <- lapply(1:(H), function(g) (eigen.list[[g]]$vectors *
                                      (1 / sqrt(interv)))[, seq(len = NPC[g]),
                                                          drop = FALSE])
  nu.hat <- lapply(1:(H), function(g){nu.hat[[g]][seq(len = NPC[g]),
                                                  drop = FALSE]})
  explained.variance <- (sigma2.hat_int + sum(unlist(nu.hat))) / total.variance

  # recalculate estimated error variance
  ######################################
  message("update error variance")
  if(H == (G + 1)){  # if no group-specific smooth errors
    newdiag <- diag(phi[[H]] %*% tcrossprod(Diagonal(length(nu.hat[[H]]),
                                                     nu.hat[[H]]), phi[[H]]))
    if(length(nu.hat[[H]]) == 0){
      newdiag <- rep(0,D)}
    sigma2.hat <- max(mean((diago - newdiag)[floor(D * 0.2):ceiling(D * 0.8)],
                           na.rm = TRUE), 0)  # avoid boundary issues
  }else{
    newdiag <- list()
    sigma2.hat_parts <- numeric()
    use <- (G + 1):H
    for(k in seq(along = use)){
      newdiag[[k]] <-
        diag(phi[[use[k]]] %*% tcrossprod(Diagonal(length(nu.hat[[use[k]]]),
                                                   nu.hat[[use[k]]]), phi[[use[k]]]))
      if(length(nu.hat[[use[k]]]) == 0){
        newdiag[[k]] <- rep(0,D)
      }
      sigma2.hat_parts[k] <- max(mean((diagos[[k]] -
                                         newdiag[[k]])[floor(D * 0.2):ceiling(D * 0.8)], na.rm = TRUE), 0)
    }
    sigma2.hat <- max(mean(sigma2.hat_parts, na.rm = TRUE), 0)
  }

  # predict basis weights
  ########################
  message("predict basis weights")
  ZtYfunc <- function(g){
    foroneq <- function(q){
      as.vector(crossprod(Zlist[[g]][[q]], Y.tilde) %*% phi[[g]][(q -
                                                                    1) * D + (1:D), ])
    }
    rowSums(sapply(1:rhovec[g], 'foroneq'))
  }
  ZtY <- do.call("c", lapply((1:(H))[NPC != 0], 'ZtYfunc'))

  ZtZcomp <- function(g, h){
    foroneqr <- function(q, r){
      phitphi <- crossprod(phi[[g]][(q - 1) * D + (1:D), ],
                           phi[[h]][(r - 1) * D + (1:D), ])
      ZtZ <- # avoid unnecessary multiplications with pseudo-Design
        # Zlist[[H]] (= Identity):
        if(H == G + 1){
          if(g == H){
            Zlist[[h]][[r]]
          }else{
            if(h == H){
              t(Zlist[[g]][[q]])
            }else{
              crossprod(Zlist[[g]][[q]], Zlist[[h]][[r]])
            }
          }
        }else{
          crossprod(Zlist[[g]][[q]], Zlist[[h]][[r]])
        }
      return(kronecker(phitphi, ZtZ))
    }

    Reduce('+',
           mapply(foroneqr,
                  c(rep(1:rhovec[g], rhovec[h])),
                  c(rep(1:rhovec[h], each = rhovec[g])),
                  SIMPLIFY = FALSE))
  }

  ZtZ <- matrixouter((1:(H))[NPC != 0], (1:(H))[NPC != 0], "ZtZcomp")

  cIN <- c(0, cumsum(Lvec * NPC))
  IN <- Lvec * NPC
  Dinv <- Diagonal(sum(IN), rep(1 / unlist(nu.hat), rep(Lvec, NPC)))
  b.hat <- solve(ZtZ + sigma2.hat * Dinv, ZtY)
  xi.hat <- vector(H, mode = "list")
  xi.hat[NPC != 0] <- lapply((1:(H))[NPC != 0],
                             function(g) matrix(b.hat[cIN[g] + 1:IN[g]], Lvec[g]))

  ###########################################################################
  # return results
  ###########################################################################
  results <- list(Y = Y, gridpoints = gridpoints, groups = groups,
                  Zvars = Zvars, rhovec = rhovec, Lvec = Lvec, NPC = NPC, phi = phi,
                  sigma2 = sigma2.hat, nu = nu.hat, xi = xi.hat, L = L,
                  totvar = total.variance, exvar = explained.variance, bf = bf,
                  smooth = smooth, smoothalg = smoothalg, Zlist = Zlist)

  return(results)
}
