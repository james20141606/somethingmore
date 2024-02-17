MA 589 Project 1
================
Yumeng Cao
2024-02-16

# 1

## (a)

About function $f(x) = \log(1 + e^x)$.

``` r
## define the log1pe function
log1pe <- function(x) {
  return(log(1 + exp(x)))
}

## prepare values for plots
x <- seq(-5, 5, length = 100)

## define the (x)_+ function
plus <- function(x) {
  return(pmax(0,x))
}

## plot two functions in the range (−5, 5)
plot(x, log1pe(x), type = "l", ylab = "f(x) / (x)_+", main = "Log1pe and (x)_+ Functions")
lines(x, plus(x), type = "l", col = "red")
legend("bottomright", legend=c("log1pe", "(x)_+"), col=c("black", "red"), lty=1:1)
```

![](project1_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

**Answer 1.(a):** From the following plot, we could notice that the
log1pe function is smooth and differentiable everywhere, especially for
x around to zero; whereas the plus funtion (x)\_+ has a sharp corner
around to zero. And the log1pe function approaches the plus function
when x becomes larger or smaller (i.e. far from the zero). So it’s
called ‘soft-plus’ because it’s the ‘soft’ version of plus function.

## (b)

``` r
log1pe(0)
```

    ## [1] 0.6931472

``` r
log1pe(-80)
```

    ## [1] 0

``` r
log1pe(80)
```

    ## [1] 80

``` r
log1pe(800)
```

    ## [1] Inf

**Answer 1.(b):**

$$ \begin{align*}
log1pe(0) &= 0.6931472 \\
log1pe(-80) &= 0 \\
log1pe(80) &= 80 \\
log1pe(800) &= Inf \\
\end{align*}$$

The $f(80) = \log(1 + e^{80})$ and $f(800) = \log(1 + e^{800})$ gives
unexpected results.  
For $f(80)$, the actual result should be larger than 80. But due to the
limitations of floating-point precision, the $e^{80}$ is too large, so
the precision ignore the value of 1 when it is added, which leading the
machine only compute $\log(e^{80})$ and simplifies it to 80.  
For $f(800)$, it gives Inf, indicating an overflow error as $e^{800}$
exceeds the range can be shown in the floating point and therefore it is
shown as infinity.

## (c)

**Answer 1.(c):** To avoid computations if $x \ll 0$, we’d like to find
a threshold $x_{threshold}$ s.t.
$$f(x)=\log(1 + e^{x})\approx0 \text{,  if } x< x_{threshold}$$
$$1+\epsilon/2==1$$ so we have $$e^{x_{threshold}}\leq\epsilon/2$$, then
$$x_{threshold}\leq\log(\epsilon/2)$$
$$\text{Then we can approximate } f(x)=\log(1 + e^{x})\approx0 \text{ when } x_{threshold}\leq\log(\epsilon/2) \text{ without computing. }$$

``` r
## update the log1pe function
log1pe <- function(x) {
  epsilon <- .Machine$double.eps
  if (x <= log(epsilon/2)) {
    return(0)
  } else {
    return(log(1 + exp(x)))
  }
}
```

## (d)

**Answer 1.(d):**
$$f(x)=\log(1 + e^{x})=\log[e^{x}(e^{-x}+1)]=\log(e^{x})+\log(1+e^{-x})=x+log(1+e^{-x}) \text{ when } x \gg 0$$
In this case, $e^{-x}$ would decrease and approach to zero as x
increases, which could avoid overflow issues. As shown in part (c), we
would

``` r
## update the log1pe function
log1pe <- function(x) {
  epsilon <- .Machine$double.eps
  if (x < log(epsilon/2)) {
    return(0)
  } else if (x > 0){
    return(x + log(1 + exp(-x)))
  } else {
    return(log(1 + exp(x)))
  }
}
```

## (e)

**Answer 1.(e):** To avoid overflows, we could use the same method in
part (d), that is:  
$$f(x, y)=\log(e^{x} + e^{y})=\log[e^{x}(1+e^{y-x})]=\log(e^{x})+\log(1+e^{y-x})=x+log(1+e^{y-x}) \text{ if } x \ge y$$
$$f(x, y)=\log(e^{x} + e^{y})=\log[e^{y}(e^{x-y}+1)]=\log(e^{y})+\log(e^{x-y}+1)=y+log(1+e^{x-y}) \text{ if } x < y$$
which can be summarized as
$$f(x, y)=\log(e^{x} + e^{y})=\log[e^{max(x,y)}(1+e^{min(x,y)-max(x,y)})]=max(x,y)+\log[(1+e^{min(x,y)-max(x,y)})]$$

``` r
## define the lse2 function
lse2 <- function(x,y) {
  return(max(x,y) + log1pe(min(x,y) - max(x,y)))
}

# lse2 <- function(x,y) {
#   if (x > y) {
#     return(x + log1pe(y - x))
#   } else {
#     return(y + log1pe(x - y))
#   }
# }
```

## (f)

$$\begin{align*}
\text{lse}(x) &= \log \sum_{i=1}^{n} e^{x_i} \\ 
&= \log(e^{x_1}+e^{x_2}+...+e^{x_n}) \\
&= \log(e^{log(e^{x_1}+e^{x_2})}+e^{x_3}+...+e^{x_n})  \\
&=\log(e^{lse2(x_1, x_2)}+e^{x_3}+...+e^{x_n}) \\
&=\log(e^{lse2(lse2(x_1, x_2), x_3)}+e^{x_4}+...+e^{x_n}) \\
&= ...... \end{align*}$$

``` r
## Define the lse function
lse <- function(x) {
  
  sum <- x[1]
  # cat("step 1 :", sum, "\n")
  
  for(i in 2:length(x)) {
    sum <- lse2(sum, x[i])
    # cat("step", i, ":", sum, "\n")
  }
  
  # cat("\nThe final results is", sum, "\n")
  return(sum)
}

lse(rep(800, 10))
```

    ## [1] 802.3026

**Answer 1.(f):** $lse(rep(800, 10)) = 802.3026$

# 2

## (a)

``` r
## initial setting
y <- 96
m1 <- 200
m2 <- 775
s1 <- 205

## define a function to compute P0
P0 <- function(theta) {
  a <- max(0, s1 - m2)
  b <- min(m1, s1)
  P0 <- sum(sapply(a:b, function(j) choose(m1, j) * choose(m2, s1 - j) * exp(theta * j)))
  return(P0)
}

P0(0)
```

    ## [1] 1.854403e+216

``` r
P0(3)
```

    ## [1] Inf

**Answer 2.(a):**

$$P_0(0) = 1.854403*10^{216} $$

$$P_0(3) = Inf $$

Yes, there is an issue with $P_0(3) = Inf$, indicating there is an
overflow error.

## (b)

let $c_j = {m_1 \choose j} {m_2 \choose s_1 - j}$, then:

$$ \begin{align*}
\log P_0(\theta) &= log \sum_{j=a}^{b} c_j e^{\theta j} \\
&= log (c_a e^{\theta a} + c_{a+1} e^{\theta (a+1)} + ... + c_{b} e^{\theta (b)}) \\
&= log (e^{\log(c_a)} e^{\theta a} + e^{\log(c_{a+1})} e^{\theta (a+1)} + ... + e^{\log(c_b)} e^{\theta b}) \\
&= log (e^{\log(c_a)+\theta a} + e^{\log(c_{a+1})+\theta (a+1)} + ... + e^{\log(c_b)+\theta b}) 
\end{align*}$$

,where
$\log c_j = log{m_1 \choose j} + \log {m_2 \choose s_1 - j} \text{ , j=a,...,b}$

So we could use the lse function to compute the $\log P_0(\theta)$.

Then,

$$\begin{align*}
\log f_{\theta}(y) &= \log \frac{{m_1 \choose y} {m_2 \choose s_1 - y} e^{\theta y}}{P_0(\theta)} \\
&=\log{m_1 \choose y} {m_2 \choose s_1 - y} e^{\theta y}-\log P_0(\theta) \\
&=\log{m_1 \choose y} + \log{m_2 \choose s_1 - y} + \log e^{\theta y} - \log P_0(\theta) \\
&=\log{m_1 \choose y} + \log{m_2 \choose s_1 - y} + \theta y - \log P_0(\theta)
\end{align*}$$

So we could compute the $\log f_{\theta}(y)$ easily.

``` r
## Define a function to compute log(P0(theta))
logP0 <- function(theta) {
  a <- max(0, s1 - m2)
  b <- min(m1, s1)
  log_terms <- sapply(a:b, function(j) log(choose(m1, j)) + log(choose(m2, s1 - j)) + theta * j)
  ## then use lse funtion to return the logP_0
  logP0 <- lse(log_terms)
  return(logP0)
}

logP0(0)
```

    ## [1] 497.9759

``` r
logP0(3)
```

    ## [1] 762.9187

``` r
## Define a function to compute log(f_theta(y))
logFtheta <- function(theta, y) {
  log_numerator <- log(choose(m1, y)) + log(choose(m2, s1 - y)) + theta * y
  log_denominator <- logP0(theta)
  logFtheta <- log_numerator - log_denominator
  return(logFtheta)
}

logFtheta(0,96)
```

    ## [1] -50.81855

``` r
logFtheta(3,96)
```

    ## [1] -27.76135

**Answer 2.(b):**

$$\begin{align*}
& \log P_0(0)=497.9759 \\
& \log P_0(3)=762.9187 \\ \\
& \log f_{3}(96)=-27.76135 \\
& \log f_{0}(96)=-50.81855 \\ \\
& \Longrightarrow f_{3}(96) > f_{0}(96)
\end{align*}$$

Therefore, $\theta = 3$ makes the observed data ($y=96$) more likely
because \$f\_{3}(96) \> f\_{0}(96) \$.

## (c)

let $c_{jk} = j^k {m_1 \choose j} {m_2 \choose s_1 - j}$, then:

$$ \begin{align*}
\log P_k(\theta) &= log \sum_{j=a}^{b} c_{jk} e^{\theta j} \\
&= log (c_{ak} e^{\theta a} + c_{(a+1)k} e^{\theta (a+1)} + ... + c_{bk} e^{\theta (b)}) \\
&= log (e^{\log(c_{ak})} e^{\theta a} + e^{\log(c_{(a+1)k})} e^{\theta (a+1)} + ... + e^{\log(c_{bk})} e^{\theta b}) \\
&= log (e^{\log(c_{ak})+\theta a} + e^{\log(c_{(a+1)k})+\theta (a+1)} + ... + e^{\log(c_{bk})+\theta b}) 
\end{align*}$$

where
$\log c_{jk} = k* log(j) + log{m_1 \choose j} + \log {m_2 \choose s_1 - j} \text{ , j=a,...,b}$

then we could compute $\mu$ and $\sigma^2$ by the following euqations to
avoid overflows.

$$\begin{align*}
\mu &= \frac {P_1(\theta)}{P_0(\theta)} \\
&=e^{(log \frac {P_1(\theta)}{P_0(\theta)})} \\
&=e^{log{P_1(\theta)}-log{P_0(\theta)})} \\\\
\sigma^2 &= \frac {P_2(\theta)}{P_0(\theta)} - \mu^2 \\
&=e^{log{P_2(\theta)}-log{P_0(\theta)})} - \mu^2 
\end{align*}$$

``` r
## Define a function to compute P_k
logP_k <- function(k, theta) {
  a <- max(0, s1 - m2)
  b <- min(m1, s1)
  
  ##if k==0, there would be no problem with k*log(j)
  if (k==0) {
    log_terms <- sapply(1:b, function(j) log(choose(m1, j)) + log(choose(m2, s1 - j)) + theta * j)
  } else {
    ## if j==0, we need to shift the equation
    if (a==0) {
      log_terms <- sapply(1:b, function(j) k * log(j) + log(choose(m1, j)) + log(choose(m2, s1 - j)) + theta * j)
    } else {
        log_terms <- sapply(a:b, function(j) k * log(j) + log(choose(m1, j)) + log(choose(m2, s1 - j)) + theta * j)
      }
  }
  ## then use lse funtion to return the logP_k
  logP_k <- lse(log_terms)
  return(logP_k)
}

logP_k(0,3)
```

    ## [1] 762.9187

``` r
logP_k(1,3)
```

    ## [1] 767.8172

``` r
logP_k(2,3)
```

    ## [1] 772.7171

``` r
mu <- exp(logP_k(1,3) - logP_k(0,3))
cat("The mean is", mu, "when θ = 3 \n")
```

    ## The mean is 134.0799 when θ = 3

``` r
sigma2 <- exp(logP_k(2,3) - logP_k(0,3)) - mu^2
cat("The variance is", sigma2, "when θ = 3 \n")
```

    ## The variance is 26.29222 when θ = 3

``` r
high <- mu + sqrt(sigma2)
low <- mu - sqrt(sigma2)
cat("The interval [mu - sigma, mu + sigma] is [", low, "," ,high, "] \n")
```

    ## The interval [mu - sigma, mu + sigma] is [ 128.9523 , 139.2075 ]

``` r
## check if y=96 fall into the interval
y_within_interval <- y >= low && y <= high
print(y_within_interval)
```

    ## [1] FALSE

``` r
## Define a function to compute CDF(y)
compute_cdf <- function(theta, y) {
  a <- max(0, s1 - m2)
  cdf <- sum(sapply(a:y, function(j) {
    exp(log(choose(m1, j)) + log(choose(m2, s1 - j)) + theta * j - logP0(theta))
  }))
  return(cdf)
}

compute_cdf(3, 96)
```

    ## [1] 1.206731e-12

**Answer 2.(c):** The mean $\mu = 134.0799$ and variance
$\sigma^2 = 26.29222$ when $\theta = 3$. And y=96 is not “typical” under
$\theta = 3$ because it didn’t fall within the interval of
$[\mu - \sigma, \mu + \sigma] = [128.9523, 139.2075]$ and its
$CDF(y=96) = 1.206731*10^{-12}$ is very close to 0 and less than 0.025.

$$CDF(y=96) = \sum_{j=a}^{96} f_{\theta}(j) = 1.206731*10^{-12}$$

## (d)

$$\begin{align*}
L(\theta | y) &= f_{\theta}(y) \\ \\
\log L(\theta | y) &= \log f_{\theta}(y) \\
&=\log{m_1 \choose y} + \log{m_2 \choose s_1 - y} + \theta y - \log P_0(\theta) \\
& \text{  (as shown in 2(b))}
\end{align*}$$

And we alreday have a function *logFtheta* to compute the log-likelihood
for a single value of theta, so here we need a new function to compute
log-likelihoods across a range of theta values.

``` r
y <- 96
m1 <- 200
m2 <- 775
s1 <- 205

# Define a function to compute log-likelihoods across a range of theta values
log_likelihoods <- function(theta_values, y) {
  log_likelihoods <- sapply(theta_values, logFtheta, y=y)
  return(log_likelihoods)
}

theta_values <- seq(0, 4, by=0.05)
log_likelihoods <- log_likelihoods(theta_values, y)

## the likelihood curve plot
plot(theta_values, log_likelihoods, type='l', col='blue', xlab="Theta", ylab="Log-Likelihood",
     main="Log-Likelihood Curve")
```

![](project1_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
## Find the MLE(theta)
mle_theta <- theta_values[which.max(log_likelihoods)]
cat("The MLE estimated theta is", mle_theta)
```

    ## The MLE estimated theta is 1.75

``` r
## Assess whether y=96 is typical for MLE(theta)=1.75
mu <- exp(logP_k(1,1.75) - logP_k(0,1.75))
cat("The mean is", mu, "when θ = 1.75 \n")
```

    ## The mean is 96.73229 when θ = 1.75

``` r
sigma2 <- exp(logP_k(2,1.75) - logP_k(0,1.75)) - mu^2
cat("The variance is", sigma2, "when θ = 1.75 \n")
```

    ## The variance is 32.57109 when θ = 1.75

``` r
high <- mu + sqrt(sigma2)
low <- mu - sqrt(sigma2)
cat("The interval [mu - sigma, mu + sigma] is [", low, "," ,high, "] \n")
```

    ## The interval [mu - sigma, mu + sigma] is [ 91.02518 , 102.4394 ]

``` r
## check if y=96 fall into the interval
y_within <- y >= low && y <= high
print(y_within)
```

    ## [1] TRUE

``` r
## calculate CDF(96) under theta=1.75
compute_cdf(1.75, 96)
```

    ## [1] 0.4828985

**Answer 2.(d):** The MLE estimate for $\theta$ is 1.75. The y=96 is
“typical” under $\theta = 1.75$ because it fall within the interval of
$[\mu - \sigma, \mu + \sigma] = [91.02518, 102.4394]$ and its
$CDF(y=96) = 0.4828985$ within the range of $[0.025, 0.975]$ at 0.05
level.

$$CDF(y=96) = \sum_{j=a}^{96} f_{\theta=1.75}(j) = 0.4828985$$

# 3

## (a)

The log-likelihood function is showing below:

$$\begin{align*}
\mathbb{P}_{\mu,\sigma^2,\rho}(X_i) &= \frac{1}{(2\pi)^{\frac{k}{2}}|\sigma^2R(\rho)|^{\frac{1}{2}}} \exp\left(-\frac{1}{2}(X_i-\mu)^T(\sigma^2R(\rho))^{-1}(X_i-\mu)\right) \\
\end{align*}$$

$$ \ell(\mu,\sigma^2,\rho;X) = \log \mathbb{P}_{\mu,\sigma^2,\rho}(X) $$

$$ \ell(\mu,\sigma^2,\rho;X) = \log \prod_{i=1}^n \mathbb{P}_{\mu,\sigma^2,\rho}(X_i)$$

$$ \ell(\mu,\sigma^2,\rho;X) = \sum_{i=1}^n \log \mathbb{P}_{\mu,\sigma^2,\rho}(X_i)$$

$$\begin{align*}
\ell(\mu,\sigma^2,\rho;X) &= \sum_{i=1}^n \frac{1}{(2\pi)^{\frac{k}{2}}|\sigma^2R(\rho)|^{\frac{1}{2}}} \exp\left(-\frac{1}{2}(X_i-\mu)^T(\sigma^2R(\rho))^{-1}(X_i-\mu)\right) \\  
&= \sum_{i=1}^n \left(-\frac{k}{2} \log(2\pi) - \frac{1}{2}\log|\sigma^2R(\rho)| - \frac{1}{2}(X_i-\mu)^T(\sigma^2R(\rho))^{-1}(X_i-\mu)\right) \\
&= -\frac{nk}{2}\log(2\pi) - \frac{n}{2}\log|\sigma^2R(\rho)| - \frac{1}{2\sigma^2}\sum_{i=1}^n (X_i-\mu)^T R(\rho)^{-1}(X_i-\mu)
\end{align*}$$

$$\begin{align*}
\text{MLE}(\mu) &= \arg \max_{\mu} \ell(\mu,\sigma^2,\rho;X) \\
&= \arg \max_{\mu} (-\frac{nk}{2}\log(2\pi) - \frac{n}{2}\log|\sigma^2R(\rho)| - \frac{1}{2\sigma^2}\sum_{i=1}^n (X_i-\mu)^TR(\rho)^{-1}(X_i-\mu)) \\
&= \arg \max_{\mu} (- \frac{1}{2\sigma^2}\sum_{i=1}^n (X_i-\mu)^TR(\rho)^{-1}(X_i-\mu)) \\
&= \arg \max_{\mu} (- \sum_{i=1}^n (X_i-\mu)^TR(\rho)^{-1}(X_i-\mu)) \\
&= \arg \min_{\mu} (\sum_{i=1}^n (X_i-\mu)^TR(\rho)^{-1}(X_i-\mu)) \\
&= \arg \min_{\mu}  S(\mu) \\
&= \hat{\mu} 
\end{align*}$$

So we have $\text{MLE}(\mu) = \hat{\mu} = \arg \min_{\mu} S(\mu)$.

## (b)

$$ \sum_{i=1}^{n} (X_i - \mu^*) = 0 $$

$$ n\mu^* = \sum_{i=1}^{n} X_i $$

$$ \mu^* = \frac{1}{n} \sum_{i=1}^{n} X_i $$

\begin{align} 
S(\mu) &= \sum_{i=1}^{n} ( X_i - \mu^* + \mu^* - \mu )^T R(\rho)^{-1} ( X_i - \mu^* + \mu^* - \mu ) \\
&= \sum_{i=1}^{n} ((X_i - \mu^*) + (\mu^* - \mu))^T R(\rho)^{-1} ((X_i - \mu^*) + (\mu^* - \mu)) \\
&= \sum_{i=1}^{n} ( (X_i - \mu^*)^T R(\rho)^{-1} (X_i - \mu^*) + (X_i - \mu^*)^T R(\rho)^{-1} (\mu^* - \mu) + (\mu^* - \mu)^T R(\rho)^{-1} (X_i - \mu^*) + (\mu^* - \mu)^T R(\rho)^{-1} (\mu^* - \mu) ) \\
&= \sum_{i=1}^{n} ( (X_i - \mu^*)^T R(\rho)^{-1} (X_i - \mu^*) + 2(X_i - \mu^*)^T R(\rho)^{-1} (\mu^* - \mu) + (\mu^* - \mu)^T R(\rho)^{-1} (\mu^* - \mu) )
\end{align}

$$ \sum_{i=1}^{n} (X_i - \mu^*)^T R(\rho)^{-1} (\mu^* - \mu) = 0 \text{, because } \sum_{i=1}^{n} (X_i - \mu^*) = 0 $$

$$\begin{align*} 
S(\mu) &= \sum_{i=1}^{n} \left( (X_i - \mu^*)^T R(\rho)^{-1} (X_i - \mu^*) + (\mu^* - \mu)^T R(\rho)^{-1} (\mu^* - \mu) \right) \\
&= \sum_{i=1}^{n} (X_i - \mu^*)^T R(\rho)^{-1} (X_i - \mu^*) + \sum_{i=1}^{n} (\mu^* - \mu)^T R(\rho)^{-1} (\mu^* - \mu)  \\
&= \sum_{i=1}^{n} (X_i - \mu^*)^T R(\rho)^{-1} (X_i - \mu^*) + n (\mu^* - \mu)^T R(\rho)^{-1} (\mu^* - \mu)
\end{align*}$$

In the $S(\mu)$ function,
$\sum_{i=1}^{n} (X_i - \mu^*)^T R(\rho)^{-1} (X_i - \mu^*)$ is fixed as
we have $\mu^* = \frac{1}{n} \sum_{i=1}^{n} X_i$. And the \$n (^\* - )^T
R()^{-1} (^\* - ) \$.

$$ \text{In the } S(\mu) \text{ function, } \sum_{i=1}^{n} (X_i - \mu^*)^T R(\rho)^{-1} (X_i - \mu^*) \text{ is fixed as we have } \mu^* = \frac{1}{n} \sum_{i=1}^{n} X_i. \\
\text{ And the } n (\mu^* - \mu)^T R(\rho)^{-1} (\mu^* - \mu) \geq 0 . $$

$$\begin{align*} 
\arg \min_{\mu} \sum_{i=1}^{n} (X_i - \mu^*)^T  R(\rho)^{-1} (X_i - \mu^*) + n (\mu^* - \mu)^T  R(\rho)^{-1}  (\mu^* - \mu)  \\
\end{align*}$$

$$\begin{align*} 
\arg \min_{\mu} \sum_{i=1}^{n}  (X_i - \mu^*)^T  R(\rho)^{-1} (X_i - \mu^*) \\
\end{align*}$$

So
$$\text{MLE}(\mu) = \hat{\mu} = \mu^* = \frac{1}{n} \sum_{i=1}^{n} X_i$$.

## (c)

## (d)

## (e)
