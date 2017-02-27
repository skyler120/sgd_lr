#set.seed(120)
num_iters_sgd = 1000;
num_iters_gd = 100;
d = 1;
w = rep(0,d+1); w2 = rep(0,d+1);

N = 1000; p = rbinom(N,1,0.5); 
x = rnorm(N*d,mean=-1,sd=sqrt(d)/2)*p + rnorm(N*d,mean=1,sd=sqrt(d)/2)*(1-p);
y = p; y[which(y==0)] = -1;
X = matrix(x,N,d, byrow=F); X = cbind(1,X);

#plot(X[,2], X[,3], pch='.')
#pos_label = X[which(y==1),]
#points(pos_label[,2], pos_label[,3],col="red", pch='.')

likelihood <- function(w) {return(1/N * sum(log(1 + exp(-y*w%*%x))))}

grad_l <- function(w, rx,ry) {
  const = 1/(1+exp(-1*ry*w%*%rx));
  gradl = -ry*rx*exp(-ry*w%*%rx);
  return(const*gradl)
}

pt <- proc.time()[3]
for(t in 1:num_iters_sgd){
  r_ind = sample.int(1:N,1); rx = X[r_ind,]; ry = y[r_ind]; 
  g = -1*grad_l(w, rx,ry)
  w = w + 1/t * g;
}
w
time_sgd = proc.time()[3] - pt

pt <- proc.time()[3]
for(t in 1:num_iters_gd){
  lambda_gl <- function(r){
    return(grad_l(w2,r[-1],r[1]))
  }
  sg = 0
  for(n in 1:N){
    sg = sg+grad_l(w2,X[n,],y[n])
  }
  w2 = w2 - 1/t * sg/N;
}
w2
time_gd = proc.time()[3] - pt
