#' Calculate Odds Ratio (OR) or Hazard Ratio (HR) with 95% CI in a formatted string
#'
#' @param model A fitted glm (logistic) or coxph (Cox) model
#' @param level Confidence level for the interval. Default = 0.95
#' @return A data frame with formatted OR (95% CI) strings
#' 
ORCI <- function(model, level = 0.95) {

  coef_estimates <- coef(model)
  OR <- exp(coef_estimates)
  
  ci <- confint(model, level = level)
  OR_lower <- exp(ci[, 1])
  OR_upper <- exp(ci[, 2])
  
  formatted <- sprintf("%.3f (%.3f – %.3f)", OR, OR_lower, OR_upper)
  result_df <- data.frame(
    Term = names(coef_estimates),
    OR_CI = formatted,
    stringsAsFactors = FALSE
  )
  
  return(result_df)
}
