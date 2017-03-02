#fixed number of iterations SGD and GD
num_iters_sgd = 1000;
num_iters_gd = 1000;
d = 2;
w = rep(0,d+1); w2 = rep(0,d+1);
eta_gd = 1; eta_sgd = 1;

likelihood <- function(w) {return(log(1 + exp(-ry*w%*%rx)))}

grad_l <- function(w, rx,ry) {return(-ry*rx*1/(1+exp(ry*w%*%rx)))}

lambda_gl <- function(r){ return(grad_l(w2,r[-1],r[1]))}

N = 1000; p = rbinom(N,1,0.5); 
x = rnorm(N*d,mean=-1,sd=sqrt(d)/2)*p + rnorm(N*d,mean=1,sd=sqrt(d)/2)*(1-p);
y = p; y[which(y==0)] = -1;
X = matrix(x,N,d, byrow=F); X = cbind(1,X);

pt <- proc.time()[3];
total_w = 0;
plot_est_ws_sgd = c();
for(t in 1:num_iters_sgd){
  r_ind = sample.int(N,1); rx = X[r_ind,]; ry = y[r_ind]; 
  g = -1*grad_l(w, rx,ry);
  #eta_sgd = 1/sqrt(t);
  w = w + eta_sgd*g;
  total_w = total_w + w;
  plot_est_ws_sgd = c(plot_est_ws_sgd, total_w/t)
}
sgd_w = total_w/num_iters_sgd;
time_sgd = proc.time()[3] - pt;

pt <- proc.time()[3]
plot_est_ws_gd = c();
for(t in 1:num_iters_gd){
  sg = rep(0, d+1);
  for(n in 1:N){
    sg = sg+grad_l(w2,X[n,],y[n]);
  }
  #eta_gd = 1/sqrt(t);
  w2 = w2 - eta_gd*sg/N;
  plot_est_ws_gd = c(plot_est_ws_gd, w2)
}
gd_w = w2;
time_gd = proc.time()[3] - pt;

pt <- proc.time()[3]
iwls = glm(p~1+X[,2]+X[,3], family="binomial", start=rep(0,d+1))
time_iwls = proc.time()[3] - pt

iwls$coefficients
sgd_w
gd_w

time_iwls
time_sgd
time_gd

plot(plot_est_ws_gd[seq(2,length(plot_est_ws_gd), 3)],plot_est_ws_gd[seq(3,length(plot_est_ws_gd), 3)], type='l', lwd=2, col="red", xlab = "theta_1", ylab = "theta_2")
#points(1:(length(plot_est_ws_sgd)/3), plot_est_ws_sgd[seq(2,length(plot_est_ws_sgd), 3)], pch=20, col="blue")

