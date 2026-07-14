#' Black-Scholes-Merton Option Pricing
#'
#' @param S Spot price
#' @param K Strike price
#' @param T Time to expiry (years)
#' @param r Risk-free rate
#' @param sigma Volatility
#' @param q Dividend yield (default 0)
#'
#' @return Option price
#' @export
bsm_call <- function(S, K, T, r, sigma, q = 0) {
  d1 <- (log(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
  d2 <- d1 - sigma * sqrt(T)
  S * exp(-q * T) * pnorm(d1) - K * exp(-r * T) * pnorm(d2)
}

#' @rdname bsm_call
#' @export
bsm_put <- function(S, K, T, r, sigma, q = 0) {
  d1 <- (log(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
  d2 <- d1 - sigma * sqrt(T)
  K * exp(-r * T) * pnorm(-d2) - S * exp(-q * T) * pnorm(-d1)
}

#' @rdname bsm_call
#' @export
bsm_delta <- function(S, K, T, r, sigma, q = 0, type = "call") {
  d1 <- (log(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
  if (type == "call") {
    exp(-q * T) * pnorm(d1)
  } else {
    exp(-q * T) * (pnorm(d1) - 1)
  }
}

#' @rdname bsm_call
#' @export
bsm_gamma <- function(S, K, T, r, sigma, q = 0) {
  d1 <- (log(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
  exp(-q * T) * dnorm(d1) / (S * sigma * sqrt(T))
}

#' @rdname bsm_call
#' @export
bsm_vega <- function(S, K, T, r, sigma, q = 0) {
  d1 <- (log(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
  S * exp(-q * T) * dnorm(d1) * sqrt(T) / 100
}
