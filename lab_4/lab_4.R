library(dplyr)
library(gsubfn)

# Main functions
inverse <- function(f, lower = -100, upper = 100) {
  function(y) {
    uniroot(function(x) {
      f(x) - y 
    }, lower = lower, upper = upper)[1]
  }
}

get.Gkr <- function(prob, f1, f2) {
  isf <- inverse(function(x) 1 - pf(x, f1, (f2 - 1) * f1))
  f_crit <- isf((1 - prob) / f2)$root
  Gkr <- f_crit / (f_crit + f2 - 1)
}

get.X_ranges <- function(X_names) {
  X_ranges <- c(X1_min, X1_max, X2_min, X2_max, X3_min, X3_max) %>% 
    array(dim = c(2, 3),
          dimnames = list(c('min', 'max'), X_names)) %>% 
    t()
}

get.Y_range <- function(X_ranges) {
  X_max_mean <- mean(X_ranges[, 2])
  X_min_mean <- mean(X_ranges[, 1])
  
  Y_max <- 200 + X_max_mean
  Y_min <- 200 + X_min_mean
  c(Y_min, Y_max)
}

get.X_norms <- function(X_names, interaction = F) {
  if (! interaction) {
    X1_norm <- c(-1, -1, 1, 1)
    X2_norm <- c(-1, 1, -1, 1)
    X3_norm <- c(-1, 1, 1, -1)
    
  } else {
    X1_norm <- c(-1, -1, -1, -1, 1, 1, 1, 1)
    X2_norm <- c(-1, -1, 1, 1, -1, -1, 1, 1)
    X3_norm <- c(-1, 1, -1, 1, -1, 1, -1, 1)
  }
  
  N <- length(X1_norm)
  
  X_norm <- c(X1_norm, X2_norm, X3_norm) %>% 
    array(dim = c(N, 3), dimnames = list(c(), X_names)) %>% 
    t()
}

get.X_abs <- function(X_norm, X_ranges, k, N, X_names) {
  X_abs <- sapply(1:k, function(i) {
    sapply(X_norm[i,], function(elem) {
      ifelse(elem == 1, X_ranges[i, 2], X_ranges[i, 1])
    })
  }) %>% array(dim = c(N, 3), dimnames = list(c(), X_names)) %>% t()
  return(X_abs)
}

get.Y_exp <- function(m, N, Y_min, Y_max) {
  Y_names <- paste('Y', 1:m, sep='')
  
  Y_exp <- runif(n = m * N, min = Y_min, max = Y_max) %>% 
    matrix(nrow = N, ncol = m, dimnames = list(c(), Y_names))
  return(Y_exp)
}

get.df <- function(X, Y, interaction) {
  l1 <- length(Y[1, ])
  matrix <- cbind(t(X), Y)
  
  df <- as.data.frame(matrix)
  l2 <- length(df[1, ])
  
  if (interaction) {
    df$X1_X2 <- df$X1 * df$X2
    df$X1_X3 <- df$X1 * df$X3
    df$X2_X3 <- df$X2 * df$X3
    df$X1_X2_X3 <- df$X1 * df$X2 * df$X3
    
    df <- df[, c('X1', 'X2', 'X3', 'X1_X2',
                 'X1_X3', 'X2_X3', 'X1_X2_X3', 
                 'Y1', 'Y2', 'Y3')]
  }
  
  df$Y_means <- rowMeans(matrix[, (l2-l1+1):l2])
  df
}

get.lm <- function(df, interaction = F) {
  ifelse(interaction, 
         return(lm(Y_means ~ X1 * X2 * X3, data = df)), 
         return(lm(Y_means ~ X1 + X2 + X3, data = df)))
}

check.lm <- function(lm, Y, newdata) {
  preds <- predict(lm, newdata)
  result <- data.frame(True = Y, Predicted = preds)
}

check.Cohren <- function(df, Y, m, N, prob) {
  f1 <- m - 1
  f2 <- N
  
  Gkr <- get.Gkr(prob, f1, f2)
  
  Y_vars <- diag(var(cbind(df$Y_means, Y)))
  
  Gp <- max(Y_vars) / sum(Y_vars)
  
  c(ifelse(Gkr > Gp, T, F), Y_vars)
}

check.Student <- function(m, N, Y_vars, Y_means, X) {
  f3 <- (m - 1) * N
  t_krit <- qt(0.975, df = f3)
  
  S2b <- mean(Y_vars) / (N * m)
  Sb <- sqrt(S2b)
  
  b <- rep(1, N)
  
  betas <- sapply(1:N, function(i) rowMeans(Y_means * cbind(b, X)[i,]))
  
  tp <- abs(betas) / Sb
  
  d <- sum(ifelse(t_krit < tp, 1, 0))
  
  print('Number of significant coefs:')
  print(d)
  
  significance <- tp > t_krit
  print(significance)
  
  list(significance, d, S2b)
}

check.Fisher <- function(m, N, d, Y, preds, S2b) {
  f3 <- (m - 1) * N
  f4 <- N - d + 1
  S2ad <- (m / f4) * sum((preds - Y) ^ 2)
  
  Fp <- S2ad / S2b
  print('Fp:')
  print(Fp)
  
  f_krit <- qf(0.95, df1 = f4, df2 = f3)
  print('Fkr:')
  print(f_krit)
  
  ifelse(Fp < f_krit, T, F)
}

check.Gohren <- function(
  X_norm, X_abs, Y_exp, X_names, interaction, m, N, prob, Y_min, Y_max
) {
  repeat {
    print('Regression for norm values:')
    list[df_norm_data, norm_regr] <- run.regression(X_norm, Y_exp, X_names, interaction)
    
    print('Regression for absolute values:')
    list[df_data, abs_regr] <- run.regression(X_abs, Y_exp, X_names, interaction)
    
    # Cohren's criteria
    print('Cohren\'s criteria:')
    list[cohren.criteria, Y_vars] <- check.Cohren(df_data, Y_exp, m, N, prob)
    
    if (cohren.criteria) {
      return(list(df_data, abs_regr, norm_regr, Y_vars, Y_exp, m))
    } else {
      print('Unstable variances! Change m = m + 1')
      m <- m + 1
      Y_exp <- cbind(Y_exp, runif(n = N, min = Y_min, max = Y_max))
      dimnames(Y_exp) <- list(paste('Y', 1:length(Y_exp[, 1]), sep=''))
    }
  }
}

run.regression <- function(X, Y, X_names, interaction) {
  # Regression:
  df <- get.df(X, Y, interaction)
  
  regression <- get.lm(df, interaction)
  print(regression)
  
  # Let's put Xs:
  results <- check.lm(regression, df$Y_means, df[X_names])
  
  print('Regression results:')
  print(results)
  
  list(df, regression)
}

run.experiment <- function(
  k = 3, m = 3, prob = 0.95, interaction = F
) {
  X_names <- c('X1', 'X2', 'X3')
  
  X_ranges <- get.X_ranges(X_names)
  print('X_ranges:')
  print(X_ranges)
  
  Y_range <- get.Y_range(X_ranges)
  Y_min <- Y_range[1]
  Y_max <- Y_range[2]
  print('Y_range:')
  print(Y_range)
  
  X_norm <- get.X_norms(X_names, interaction)
  print('X_norm:')
  print(X_norm)
  
  N <- length(X_norm[1, ])
  
  X_abs <- get.X_abs(X_norm, X_ranges, k, N, X_names)
  print('X_abs:')
  print(X_abs)
  
  Y_exp <- get.Y_exp(m, N, Y_min, Y_max)
  print('Y_exp:')
  print(Y_exp)
  
  list[df_data, abs_regr, norm_regr, Y_vars, Y_exp, m] <- check.Gohren(
    X_norm, X_abs, Y_exp, X_names, interaction, m, N, prob, Y_min, Y_max
  )
  print('Stable variances!')
  
  print('Student criteria:')
  list[significance, d, S2b] <- check.Student(
    m, N, Y_vars, df_data$Y_means, 
    df_data[, !names(df_data) %in% c('Y1', 'Y2', 'Y3', 'Y_means')])
  
  # Deleting unsignificant coefs
  abs_regr$coefficients <- abs_regr$coefficients * significance 
  
  print('After deleting unsignificant coefs:')
  print(abs_regr)
  
  check.lm(abs_regr, df_data$Y_means, df_data[X_names])
  
  print('Fisher criteria:')
  
  preds <- predict(abs_regr, df_data[X_names])
  f.criteria <- check.Fisher(m, N, d, df_data$Y_means, preds, S2b)
  
  if (isTRUE(f.criteria)) {
    print('Regression is adequate to original function!')
    return(T)
  } else {
    print('Regression is unadequate to original function!')
    return(F)
  }
}

# Variant 209
X1_min <- -30
X1_max <- 0
X2_min <- 10
X2_max <- 60
X3_min <- 10
X3_max <- 35

# Let's run experiments
num_tries <- 1

# Without interaction effect
run.experiment()

# With interacti  on effect
for (try in 1:num_tries) {
  run.experiment(interaction = T)
}  