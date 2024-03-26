functions {
  // This is the PCAR distribution where
  // tau is the conditional precision;
  // alpha is the constant that make the precision matrix proper (i.e. rho in the paper)
  // d is the vector of neighbours
  real sparse_car_lpdf(vector phi, real tau, real alpha, 
    int[,] W_sparse, vector d, vector eigens, int N, int N_adj_pairs) {
      row_vector[N] phit_D; // phi' * D
      row_vector[N] phit_W; // phi' * W
      vector[N] ldet_terms;
    
      phit_D = (phi .* d)';
      phit_W = rep_row_vector(0, N);
      for (i in 1:N_adj_pairs) {
        phit_W[W_sparse[i, 1]] = phit_W[W_sparse[i, 1]] + phi[W_sparse[i, 2]];
        phit_W[W_sparse[i, 2]] = phit_W[W_sparse[i, 2]] + phi[W_sparse[i, 1]];
      }
    
      for (i in 1:N) ldet_terms[i] = log1m(alpha * eigens[i]);
      return 0.5 * (N * log(tau)
                    + sum(ldet_terms)
                    - tau * (phit_D * phi - alpha * (phit_W * phi)));
  }
}

data {
  int<lower=1> N; // number of areas
  int<lower=1> TT; // number of time points
  int<lower=1> NT; // number of areas*number of time points
  vector<lower=1, upper=N>[N] d;  // vector of number of neighbours
  matrix<lower=0, upper=1>[N,N] W; // matrix of spatial weights
  vector[N] zeros;
  int<lower=1> N_adj_pairs; // Number of pairs of neighbours
  real<lower=0> scaling_factor_kappa; // scales the variance of the log - kappa's
  int<lower=0> y[NT]; // long vector of cases ordered by time: (y_11, ..., y_n1, ..., y_1T, ..., y_nT)
  vector[NT] log_E; // Offset in the log scale
  int<lower=1> p;
  matrix[NT,p] X; // covariates
}

transformed data {
  int W_sparse[N_adj_pairs, 2];   // adjacency pairs
  vector[N] eigens;       // eigenvalues of invsqrtD * W * invsqrtD
  
  { // generate sparse representation for W
  int counter;
  counter = 1;
  // loop over upper triangular part of W to identify neighbor pairs
    for (i in 1:(N - 1)) {
      for (j in (i + 1):N) {
        if (W[i, j] == 1) {
          W_sparse[counter, 1] = i;
          W_sparse[counter, 2] = j;
          counter = counter + 1;
        }
      }
    }
  }
  {
    vector[N] invsqrtD;  
    for (i in 1:N) invsqrtD[i] = 1 / sqrt(d[i]);
    
    eigens = eigenvalues_sym(quad_form(W, diag_matrix(invsqrtD)));
  }
}

parameters {
  real beta0; // intercept
  vector[p] beta; // regression parameters
  real<lower=0> nu;  // hyperparameter for kappa
  real<lower=0> sigma; // conditional std deviation 
  real<lower=0, upper=1> lambda; // spatial dependence parameter
  vector[N] kappa_log; // scaling mixture components -- log scale
  vector[NT] s; // spatio-temporal effects ordered by time: (s_11, ..., s_n1, ..., s_1T, ..., s_nT)
}

transformed parameters {
 matrix[N,N] PrecMat; 
 vector<lower=0>[N] kappa = exp(kappa_log * sqrt(nu/scaling_factor_kappa) - nu/2.0); // Scaling mixture components -- correct scale
 
 // Precision matrix for the proposed model:
 PrecMat = (1/sigma^2)*(diag_matrix(kappa .* (1-lambda + lambda*d)) - lambda * W .* (kappa*(kappa')));
}

model {
  y ~ poisson_log(log_E + beta0 + X*beta + s);
  
  // latent effects at time 1
  s[1:N] ~ multi_normal_prec(zeros, PrecMat);
  // soft sum-to-zero constraint to identify the intercept:
  sum(s[1:N]) ~ normal(0, 0.001 * N); 
  for(t in 2:TT){
    // latent effects at time 2, ..., T 
    s[((t-1)*N+1):((t-1)*N+N)] ~ multi_normal_prec(s[((t-2)*N+1):((t-2)*N+N)], PrecMat);
  }
  
  beta0 ~ normal(0.0, 1.0);
  beta ~ normal(0.0, 1.0);
  nu ~ exponential(1.0/0.3);
  sigma ~ normal(0.0,0.1);
  lambda ~ uniform(0.0,1.0);
  kappa_log ~ sparse_car(1, 0.99, W_sparse, d, eigens, N, N_adj_pairs);
}
