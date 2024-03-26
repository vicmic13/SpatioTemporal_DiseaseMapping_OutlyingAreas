setwd(".")

#  PACKAGES & FUNCTIONS  ----
library(tidyverse) # for left_join: add data to geo data
library(rstan)
options(mc.cores = parallel::detectCores())

# SHAPEFILES + DATA ----
numneigh=c(read.table("./Datasets/France_numneigh"))$x
neigh=c(read.table("./Datasets/France_neigh"))$x

# About Corsica: 
# 20th department: Corse du sud ; 96th department: Haute-Corse --> w_{20, 96} = w_{96, 20} = 1
# Additionally: w_{6, 20} = w_{20, 6} = w_{13, 20} = w_{20, 13} = w_{83, 20} = w_{20, 83} = 1
# That is, both Corsican departments are linked to Alpes maritimes (6), Bouches du Rhône (13) and Var(83) because of the daily ferries

N = length(numneigh)

W=matrix(0, N, N)
count=1
for(i in 1:N){
  for(j in 1:numneigh[i]){
    W[i,neigh[count]]=1
    count=count+1
  }
}

Q = diag(apply(W,2,sum)) - W ; I = diag(1, N, N)

# Scaling factor based on PROPER graph
Q_kappa = diag(apply(W,2,sum)) - 0.99*W # precision Q_rho with rho=0.99
Q_kappa_inv = solve(Q_kappa)
scaling_factor_kappa = exp(mean(log(diag(Q_kappa_inv))))

# Data : make sure sorted by time and then by area code (y_11, ..., y_1n, ..., yT1, ..., y_Tn)
load("./Datasets/France_Covid_Wave2_26Weeks.RData")

TT = length(unique(data_df$time)) ; NT = N*TT

X = as.matrix(data_df %>% select(older75)) ; p=ncol(X)
y = data_df %>% pull(y)
log_E = log(data_df %>% pull(E))

# FIT PROPOSED MODELS WITH RSTAN ----
n_iter = 20000 ; n_warm = 10000 ; n_thin = 10

## Heavy Rushworth with alpha=1 and independent kappa's -- HR(1) ----
Covid_HR1_Fr = stan("./Priors/HeavyRush.stan",
                    data=list(N=N, TT=TT, NT=length(y),
                              d=numneigh, W=W, zeros=rep(0,N),
                              y=y, log_E=log_E,
                              p=p, X=X),
                    init = list(list(beta0=0, beta=rep(0,p), s=rep(0, N*TT), nu=4, kappa=rep(1, N), sigma=0.25, lambda=0.95),
                                list(beta0=0, beta=rep(0,p), s=rep(0, N*TT), nu=4, kappa=rep(1, N), sigma=0.25, lambda=0.95)),
                    control=list(adapt_delta = 0.95),
                    warmup=n_warm, iter=n_iter, chains=2, thin=n_thin)

traceplot(Covid_HR1_Fr, pars = c("beta0"))

## Heavy Rushworth with alpha ~ U(-1,1) and independent kappa's -- HR(alpha) ----
Covid_HRa_Fr = stan("./Priors/HeavyRush_WithAlpha.stan",
                     data=list(N=N, TT=TT, NT=length(y),
                               d=numneigh, W=W, zeros=rep(0,N),
                               y=y, log_E=log_E,
                               p=p, X=X),
                     init = list(list(beta0=0, beta=rep(0,p), s=rep(0, N*TT), nu=4, kappa=rep(1, N), sigma=0.25, lambda=0.95, alpha=0),
                                 list(beta0=0, beta=rep(0,p), s=rep(0, N*TT), nu=4, kappa=rep(1, N), sigma=0.25, lambda=0.95, alpha=0)),
                     control=list(adapt_delta = 0.95),
                     warmup=n_warm, iter=n_iter, chains=2, thin=n_thin)

traceplot(Covid_HRa_Fr, pars = c("alpha"))

## Heavy Rushworth with alpha=1 and spatially structured kappa's -- HR-LPC(1) ----
Covid_HR1LPC_Fr = stan("./Priors/HeavyRush_logPCAR.stan", 
                       data=list(N=N, TT=TT, NT=length(y), 
                                 d=numneigh, W=W, zeros=rep(0,N),
                                 N_adj_pairs = length(neigh)/2, scaling_factor_kappa = scaling_factor_kappa,
                                 y=y, log_E=log_E,
                                 p=p, X=X), 
                       init = list(list(beta0=0, beta=rep(0,p), s=rep(0, N*TT), nu=0.3, kappa_log=rep(log(1), N), sigma=0.25, lambda=0.95),
                                   list(beta0=0, beta=rep(0,p), s=rep(0, N*TT), nu=0.3, kappa_log=rep(log(1), N), sigma=0.25, lambda=0.95)),
                       control=list(adapt_delta = 0.95),
                       warmup=n_warm, iter=n_iter, chains=2, thin=n_thin)

traceplot(Covid_HR1LPC_Fr, pars = c("lambda"))

## Heavy Rushworth with alpha ~ U(-1,1) and spatially structured kappa's -- HR-LPC(alpha) ----
Covid_HRaLPC_Fr = stan("./Priors/HeavyRush_WithAlpha_logPCAR.stan", 
                        data=list(N=N, TT=TT, NT=length(y), 
                                  d=numneigh, W=W, zeros=rep(0,N),
                                  N_adj_pairs = length(neigh)/2, scaling_factor_kappa = scaling_factor_kappa,
                                  y=y, log_E=log_E,
                                  p=p, X=X), 
                        init = list(list(beta0=0, beta=rep(0,p), s=rep(0, N*TT), nu=0.3, kappa_log=rep(log(1), N), sigma=0.25, lambda=0.95, alpha=0),
                                    list(beta0=0, beta=rep(0,p), s=rep(0, N*TT), nu=0.3, kappa_log=rep(log(1), N), sigma=0.25, lambda=0.95, alpha=0)),
                        control=list(adapt_delta = 0.95),
                        warmup=n_warm, iter=n_iter, chains=2, thin=n_thin)

traceplot(Covid_HRaLPC_Fr, pars = c("s[97]"))




