#vary sample size
convergence_error = 10^(-10);
d = 2;
w = rep(0,d+1); w2 = rep(0,d+1);
eta_sgd = 0.1; eta_gd = 0.1;
Ns = c(10,100,500,1000,5000,10000)

sgd_vals = c(); gd_vals = c(); sgd_times = c(); gd_times = c(); irls_times = c();
sgd_iters = c(); gd_iters = c(); iwls_vals = c(); iwls_iters = c();

norm_vec <- function(x) {sqrt(sum(x^2))}

likelihood <- function(w) {return(log(1 + exp(-ry*w%*%rx)))}

grad_l <- function(w, rx,ry) {return(-ry*rx*1/(1+exp(ry*w%*%rx)))}

lambda_gl <- function(r){ return(grad_l(w2,r[-1],r[1]))}

for(j in 1:length(Ns)){
  print(j)
  N = Ns[j]; p = rbinom(N,1,0.5); 
  x = rnorm(N*d,mean=-1,sd=sqrt(d)/2)*p + rnorm(N*d,mean=1,sd=sqrt(d)/2)*(1-p);
  y = p; y[which(y==0)] = -1;
  X = matrix(x,N,d, byrow=F); X = cbind(1,X)
  
  pt <- proc.time()[3];
  prev_w = rep(-(10^10),d+1);
  num_iters_sgd = 0;
  total_w = 0;
  while(norm_vec(w - prev_w) > convergence_error){
    prev_w = w;
    r_ind = sample.int(N,1); rx = X[r_ind,]; ry = y[r_ind]; 
    g = -1*grad_l(w, rx,ry);
    #w = w + eta_sgd*g;
    w = w + 1/sqrt(num_iters_sgd+1)*g;
    total_w = total_w + w
    num_iters_sgd = num_iters_sgd + 1;
  }
  sgd_w = total_w/num_iters_sgd;
  time_sgd = proc.time()[3] - pt;
  sgd_vals = c(sgd_vals, norm_vec(sgd_w + 8/d));
  sgd_iters = c(sgd_iters, num_iters_sgd);
  sgd_times = c(sgd_times, time_sgd);
  
  pt <- proc.time()[3]
  prev_w2 = rep(-(10^10),d+1);
  num_iters_gd = 0;
  while(norm_vec(w2 - prev_w2) > convergence_error){
    prev_w2 = w2;
    sg = rep(0, d+1);
    for(n in 1:N){
      sg = sg+grad_l(w2,X[n,],y[n]);
    }
    #w2 = w2 - eta_gd*sg/N;
    w2 = w2 - 1/sqrt(num_iters_gd+1)*sg/N;
    num_iters_gd = num_iters_gd + 1;
  }
  gd_w = w2;
  time_gd = proc.time()[3] - pt;
  gd_vals = c(gd_vals, norm_vec(gd_w + 8/d));
  gd_iters = c(gd_iters, num_iters_gd);
  gd_times = c(gd_times, time_gd);
  
  pt <- proc.time()[3]
  #iwls = glm(p~1+X[,2] + X[,3], family="binomial", start=rep(0,d+1))
  #time_iwls = proc.time()[3] - pt
  #iwls_vals = c(iwls_vals, norm_vec(iwls$coefficients + 8/d));
  #iwls_iters = c(iwls_iters, num_iters_gd);
  #iwls_times = c(iwls_times, time_gd);
}

