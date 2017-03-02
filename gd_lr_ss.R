#vary sample size
convergence_error = 10^(-10);
d = 2;
w = rep(0,d+1); w2 = rep(0,d+1);
eta_sgd = 0.4; eta_gd = 1;
Ns = c(100,200,500, 750,1000, 2500, 5000, 7500, 10000);
N = Ns[length(Ns)]; pp = rbinom(N,1,0.5); 
x = rnorm(N*d,mean=-1,sd=sqrt(d)/2)*pp + rnorm(N*d,mean=1,sd=sqrt(d)/2)*(1-pp);
yy = pp; yy[which(yy==0)] = -1;
W = matrix(x,N,d, byrow=F); W = cbind(1,W)

sgd_vals = c(); gd_vals = c(); sgd_times = c(); gd_times = c(); iwls_times = c();
sgd_iters = c(); gd_iters = c(); iwls_vals = c(); iwls_iters = c();

norm_vec <- function(x) {sqrt(sum(x^2))}

likelihood <- function(w) {return(log(1 + exp(-ry*w%*%rx)))}

grad_l <- function(w, rx,ry) {return(-ry*rx*1/(1+exp(ry*w%*%rx)))}

lambda_gl <- function(r){ return(grad_l(w2,r[-1],r[1]))}

for(j in 1:length(Ns)){
  print(Ns[j])
  #N = Ns[j]; p = rbinom(N,1,0.5); 
  #x = rnorm(N*d,mean=-1,sd=sqrt(d)/2)*p + rnorm(N*d,mean=1,sd=sqrt(d)/2)*(1-p);
  #y = p; y[which(y==0)] = -1;
  #X = matrix(x,N,d, byrow=F); X = cbind(1,X)
  N = Ns[j];
  X = W[1:N,]; y = yy[1:N]; p = pp[1:N];
  
  pt <- proc.time()[3];
  prev_w = rep(-(10^10),d+1);
  num_iters_sgd = 1000;
  total_w = 0;
  plot_est_ws_sgd = c();
  for(t in 1:num_iters_sgd){
    r_ind = sample.int(N,1); rx = X[r_ind,]; ry = y[r_ind]; 
    g = -1*grad_l(w, rx,ry);
    #eta_sgd = 1/sqrt(t);
    w = w + eta_sgd*g;
    total_w = total_w + w;
    #plot_est_ws_sgd = c(plot_est_ws_sgd, total_w/t)
  }
  sgd_w = total_w/num_iters_sgd;
  time_sgd = proc.time()[3] - pt;
  sgd_vals = c(sgd_vals, norm_vec(sgd_w + c(0, 8/d, 8/d)));
  #sgd_iters = c(sgd_iters, num_iters_sgd);
  sgd_times = c(sgd_times, time_sgd);
  
  pt <- proc.time()[3]
  prev_w2 = rep(-(10^10),d+1);
  num_iters_gd = 1000;
  for(t in 1:num_iters_gd){
    sg = rep(0, d+1);
    for(n in 1:N){
      sg = sg+grad_l(w2,X[n,],y[n]);
    }
    #eta_gd = 1/sqrt(t);
    w2 = w2 - eta_gd*sg/N;
    #plot_est_ws_gd = c(plot_est_ws_gd, w2)
  }
  gd_w = w2;
  time_gd = proc.time()[3] - pt;
  gd_vals =  c(gd_vals, norm_vec(gd_w + c(0, 8/d, 8/d)));
  #gd_iters = c(gd_iters, num_iters_gd);
  gd_times = c(gd_times, time_gd);
  
  pt <- proc.time()[3]
  iwls = glm(p~1+X[,2] + X[,3], family="binomial", start=rep(0,d+1))
  time_iwls = proc.time()[3] - pt
  iwls_vals = c(iwls_vals, norm_vec(iwls$coefficients + c(0, 8/d, 8/d)));
  iwls_iters = c(iwls_iters, iwls$iter);
  iwls_times = c(iwls_times, time_iwls);
}

plot(Ns, iwls_times, type='l', col="green", lwd=2, xlab = "Sample Size", ylab = "Elapsed Time")
lines(Ns, sgd_times, col="blue", lwd=2)
lines(Ns, gd_times, col="red", lwd=2)
plot(Ns, gd_times, col="red", lwd=2, xlab = "Sample Size", ylab = "Elapsed Time", type='l')
