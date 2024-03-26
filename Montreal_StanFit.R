setwd(".")

# PACKAGES & FUNCTIONS  ----
library(tidyverse) # for left_join: add data to geo data
library(rstan)
options(mc.cores = parallel::detectCores())

adjlist = function(W,N){ 
  adj=0
  for(i in 1:N){  
    for(j in 1:N){  
      if(W[i,j]==1){adj = append(adj,j)}
    }
  }
  adj = adj[-1]
  return(adj)
}

# SHAPEFILES + DATA ----
# Adjacency
W = as.matrix(read.csv("./Datasets/Montreal_Adjacency.csv", header=T, row.names=1, sep=","))

N=nrow(W)

neigh = adjlist(W, N)
numneigh = apply(W,2,sum)

Q = diag(apply(W,2,sum)) - W ; I = diag(1, N, N)

# Scaling factor based on PROPER graph
Q_kappa = diag(apply(W,2,sum)) - 0.99*W # precision Q_rho with rho=0.99
Q_kappa_inv = solve(Q_kappa)
scaling_factor_kappa = exp(mean(log(diag(Q_kappa_inv))))

# Data : make sure sorted by time and then by area code (y_11, ..., y_1n, ..., yT1, ..., y_Tn)
load("./Datasets/Montreal_Covid_Wave2_30Weeks.RData")

TT = length(unique(data_df$time)) ; NT = N*TT

X = as.matrix(data_df %>% select(diploma, age, beds)) ; p=ncol(X)
y = data_df %>% pull(y)
log_E = log(data_df %>% pull(E))

# FIT PROPOSED MODELS WITH RSTAN ----
n_iter = 20000 ; n_warm = 10000 ; n_thin = 10

## Heavy Rushworth with alpha=1 and independent kappa's -- HR(1) ----
Covid_HR1_Mtl = stan("./Priors/HeavyRush.stan",
                     data=list(N=N, TT=TT, NT=length(y),
                               d=numneigh, W=W, zeros=rep(0,N),
                               y=y, log_E=log_E,
                               p=p, X=X),
                     init = list(list(beta0=0, beta=rep(0,p), s=rep(0, N*TT), nu=4, kappa=rep(1, N), sigma=0.25, lambda=0.95),
                                 list(beta0=0, beta=rep(0,p), s=rep(0, N*TT), nu=4, kappa=rep(1, N), sigma=0.25, lambda=0.95)),
                     control=list(adapt_delta = 0.95),
                     warmup=n_warm, iter=n_iter, chains=2, thin=n_thin)

traceplot(Covid_HR1_Mtl, pars = c("beta0"))

## Heavy Rushworth with alpha ~ U(-1,1) and independent kappa's -- HR(alpha) ----
Covid_HRa_Mtl = stan("./Priors/HeavyRush_WithAlpha.stan",
                                data=list(N=N, TT=TT, NT=length(y),
                                          d=numneigh, W=W, zeros=rep(0,N),
                                          y=y, log_E=log_E,
                                          p=p, X=X),
                                init = list(list(beta0=0, beta=rep(0,p), s=rep(0, N*TT), nu=4, kappa=rep(1, N), sigma=0.25, lambda=0.95, alpha=0),
                                            list(beta0=0, beta=rep(0,p), s=rep(0, N*TT), nu=4, kappa=rep(1, N), sigma=0.25, lambda=0.95, alpha=0)),
                                control=list(adapt_delta = 0.95),
                                warmup=n_warm, iter=n_iter, chains=2, thin=n_thin)

traceplot(Covid_HRa_Mtl, pars = c("alpha"))

## Heavy Rushworth with alpha=1 and spatially structured kappa's -- HR-LPC(1) ----
Covid_HR1LPC_Mtl = stan("./Priors/HeavyRush_logPCAR.stan", 
                        data=list(N=N, TT=TT, NT=length(y), 
                                  d=numneigh, W=W, zeros=rep(0,N),
                                  N_adj_pairs = length(neigh)/2, scaling_factor_kappa = scaling_factor_kappa,
                                  y=y, log_E=log_E,
                                  p=p, X=X), 
                        init = list(list(beta0=0, beta=rep(0,p), s=rep(0, N*TT), nu=0.3, kappa_log=rep(log(1), N), sigma=0.25, lambda=0.95),
                                    list(beta0=0, beta=rep(0,p), s=rep(0, N*TT), nu=0.3, kappa_log=rep(log(1), N), sigma=0.25, lambda=0.95)),
                        control=list(adapt_delta = 0.95),
                        warmup=n_warm, iter=n_iter, chains=2, thin=n_thin)

traceplot(Covid_HR1LPC_Mtl, pars = c("kappa[1]"))

## Heavy Rushworth with alpha ~ U(-1,1) and spatially structured kappa's -- HR-LPC(alpha) ----
Covid_HRaLPC_Mtl = stan("./Priors/HeavyRush_WithAlpha_logPCAR.stan", 
                                            data=list(N=N, TT=TT, NT=length(y), 
                                                      d=numneigh, W=W, zeros=rep(0,N),
                                                      N_adj_pairs = length(neigh)/2, scaling_factor_kappa = scaling_factor_kappa,
                                                      y=y, log_E=log_E,
                                                      p=p, X=X), 
                                            init = list(list(beta0=0, beta=rep(0,p), s=rep(0, N*TT), nu=0.3, kappa_log=rep(log(1), N), sigma=0.25, lambda=0.95, alpha=0),
                                                        list(beta0=0, beta=rep(0,p), s=rep(0, N*TT), nu=0.3, kappa_log=rep(log(1), N), sigma=0.25, lambda=0.95, alpha=0)),
                                            control=list(adapt_delta = 0.95),
                                            warmup=n_warm, iter=n_iter, chains=2, thin=n_thin)

traceplot(Covid_HRaLPC_Mtl, pars = c("kappa[1]"))



