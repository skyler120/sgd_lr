###### Computes Best Values of learning rate via cross validation
###### Specify range of learning rates in eta_sgd and eta_gd
convergence_error = 10^(-10);
d = 2;
w = rep(0,d+1); w2 = rep(0,d+1);
eta_sgd = seq(0.1,1,by=0.1); eta_gd = seq(0.1,1,by=0.1);

sgd_vals = c(); gd_vals = c(); sgd_times = c(); gd_times = c(); iwls_times = c();
sgd_iters = c(); gd_iters = c(); iwls_vals = c(); 

norm_vec <- function(x) sqrt(sum(x^2))

likelihood <- function(w) {return(log(1 + exp(-ry*w%*%rx)))}

grad_l <- function(w, rx,ry) {return(-ry*rx*1/(1+exp(ry*w%*%rx)))}

lambda_gl <- function(r){ return(grad_l(w2,r[-1],r[1]))}
N = 1000; p = rbinom(N,1,0.5); 
x = rnorm(N*d,mean=-1,sd=sqrt(d)/2)*p + rnorm(N*d,mean=1,sd=sqrt(d)/2)*(1-p);
y = p; y[which(y==0)] = -1;
X = matrix(x,N,d, byrow=F); X = cbind(1,X);

for(j in 1:length(eta_sgd)){
  print(j)
  pt <- proc.time()[3];
  prev_w = rep(-(10^10),d+1);
  num_iters_sgd = 1000;
  total_w = 0;
  for(k in 1:num_iters_sgd){
  #while(norm_vec(w - prev_w) > convergence_error){
    prev_w = w;
    r_ind = sample.int(N,1); rx = X[r_ind,]; ry = y[r_ind]; 
    g = -1*grad_l(w, rx,ry);
    w = w + eta_sgd[j]*g;
    total_w = total_w + w;
    #num_iters_sgd = num_iters_sgd + 1;
  }
  sgd_w = total_w/num_iters_sgd;
  time_sgd = proc.time()[3] - pt;
  
  pt <- proc.time()[3]
  prev_w2 = rep(-(10^10),d+1);
  num_iters_gd = 1000;
  for(k in 1:num_iters_gd){
  #while(norm_vec(w2 - prev_w2) > convergence_error){
    prev_w2 = w2;
    sg = rep(0, d+1);
    for(n in 1:N){
      sg = sg+grad_l(w2,X[n,],y[n]);
    }
    w2 = w2 - eta_gd[j]*sg/N
    #num_iters_gd  = num_iters_gd + 1;
  }
  gd_w = w2
  time_gd = proc.time()[3] - pt;
  
  pt <- proc.time()[3]
  iwls = glm(p~1+X[,2] + X[,3], family="binomial", start=rep(0,d+1))
  time_iwls = proc.time()[3] - pt
  
  sgd_vals = c(sgd_vals, norm_vec(sgd_w + c(0, 8/d, 8/d)));
  gd_vals =  c(gd_vals, norm_vec(gd_w + c(0, 8/d, 8/d)));
  iwls_vals = c(iwls_vals, norm_vec(iwls$coefficients + c(0, 8/d, 8/d)));
  sgd_times = c(sgd_times, time_sgd);
  gd_times = c(gd_times, time_gd);
  iwls_times = c(iwls_times, time_iwls);
}

