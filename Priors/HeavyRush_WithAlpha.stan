data {
  int<lower=1> N; // number of areas
  int<lower=1> TT; // number of time points
  int<lower=1> NT; // number of areas*number of time points
  vector<lower=1, upper=N>[N] d;  // vector of number of neighbours
  matrix<lower=0, upper=1>[N,N] W; // matrix of spatial weights
  vector[N] zeros;
  int<lower=0> y[NT];  // long vector of cases ordered by time: (y_11, ..., y_n1, ..., y_1T, ..., y_nT)
  vector[NT] log_E;           // Offset in the log scale
  int<lower=1> p;
  matrix[NT,p] X; // covariates
}

parameters {
  real beta0; // intercept
  vector[p] beta; // regression parameters
  real<lower=0> sigma;     // conditional std deviation 
  real<lower=0, upper=1> lambda; // spatial dependence parameter
  real<lower=-1, upper=1> alpha; // temporal dependence parameter 
  vector<lower=0>[N] kappa;      // scaling mixture components
  real<lower=0> nu;        // hyperparameter for kappa
  vector[NT] s;  // spatio-temporal effects ordered by time: (s_11, ..., s_n1, ..., s_1T, ..., s_nT)
}

transformed parameters {
 // Precision matrix for the proposed model:
 matrix[N,N] PrecMat = (1/sigma^2)*(diag_matrix(kappa .* (1-lambda + lambda*d)) - lambda * W .* (kappa*(kappa')));
}

model {
  y ~ poisson_log(log_E + beta0 + X*beta + s);
  
  // latent effects at time 1
  s[1:N] ~ multi_normal_prec(zeros, PrecMat);
  // soft sum-to-zero constraint to identify the intercept:
  sum(s[1:N]) ~ normal(0, 0.001 * N); 
  for(t in 2:TT){
    // latent effects at time 2, ..., T 
    s[((t-1)*N+1):((t-1)*N+N)] ~ multi_normal_prec(alpha*s[((t-2)*N+1):((t-2)*N+N)], PrecMat);
  }
  
  beta0 ~ normal(0.0, 1.0);
  beta ~ normal(0.0, 1.0);
  nu ~ exponential(1.0/4.0);
  sigma ~ normal(0.0,0.1);
  lambda ~ uniform(0.0,1.0);
  kappa ~ gamma(nu/2.0, nu/2.0);
  alpha ~ uniform(-1.0,1.0);
}
