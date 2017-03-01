#fixed number of iterations SGD and GD
num_iters_sgd = 10000;
num_iters_gd = 500;
d = 4;
w = rep(0,d+1); w2 = rep(0,d+1);
eta_sgd = seq(0,1,by=0.01); eta_gd = seq(0,1,by=0.01);

likelihood <- function(w) {return(log(1 + exp(-ry*w%*%rx)))}

grad_l <- function(w, rx,ry) {return(-ry*rx*1/(1+exp(ry*w%*%rx)))}

N = 1000; p = rbinom(N,1,0.5); 
x = rnorm(N*d,mean=-1,sd=sqrt(d)/2)*p + rnorm(N*d,mean=1,sd=sqrt(d)/2)*(1-p);
y = p; y[which(y==0)] = -1;
X = matrix(x,N,d, byrow=F); X = cbind(1,X);

pt <- proc.time()[3];
for(t in 1:num_iters_sgd){
  r_ind = sample.int(N,1); rx = X[r_ind,]; ry = y[r_ind]; 
  g = -1*grad_l(w, rx,ry);
  w = w + eta_sgd*g;
}
time_sgd = proc.time()[3] - pt;

pt <- proc.time()[3]
for(t in 1:num_iters_gd){
  lambda_gl <- function(r){
    return(grad_l(w2,r[-1],r[1]))
  }
  sg = rep(0, d+1);
  for(n in 1:N){
    sg = sg+grad_l(w2,X[n,],y[n]);
  }
  w2 = w2 - eta_gd*sg/N
}
time_gd = proc.time()[3] - pt;

pt <- proc.time()[3]
iwls = glm(p~1+X[,2], family="binomial", start=c(0,0))
time_iwls = proc.time()[3] - pt

iwls$coefficients
w
w2

time_iwls
time_sgd
time_gd