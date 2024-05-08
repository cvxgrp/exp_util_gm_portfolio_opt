# exp_util_gm_portfolio_opt
Exponential utility and entropic value at risk (EVaR) portfolio construction under a Gaussian mixture model of returns in Python. 
This repository implements the exponential utility and EVaR objective portfolio optimization problems described in the paper [Portfolio construction with Gaussian mixture returns and exponential utility via convex optimization](https://web.stanford.edu/~boyd/papers/exp_util_gm_ret_portfolio.html).


## Optimizing a portfolio with the exponential utility
To run a minimal example, simply load example problem data defined in problem_data.py via

```from problem_data import mus, Sigmas, pi, n, k```.

This defines ```mus```, ```Sigmas```, and ```pi``` as lists of the respective mixture component means, covariance matrices, and mixture weights of a Gaussian mixture return model with 4 components, for an 11 dimensional return distribution. The, the following code from section 3.2 in the paper  solves the EGM portfolio construcion problem:

```
import cvxpy as cvx

gamma = 1

def K(w):
    u = cvx.vstack([cvx.log(pi[i])
      - gamma * mus[i] @ w
      + (gamma**2/2) * cvx.quad_form(w, Sigmas[i]) for i in range(k)])
    return cvx.log_sum_exp(u)

w = cvx.Variable(n)
objective = cvx.Minimize(K(w))
constraints = [ w >= 0, cvx.sum(w) == 1 ]
egm_prob = cvx.Problem(objective, constraints)
egm_prob.solve()
w.value
```

## Optimizing a portfolio with EVaR

After loading the EVaR portfolio optimization code via  ```from gm_evar_portfolio import min_EVaR_portfolio```
and with ```L``` the leverage limit and ```alpha``` the EVaR level, simply run

```w,delta,evar = min_EVaR_portfolio(alpha,L,mus,Sigmas,pi)``` 

to generate the minimum EVaR portfolio. 

## Citing
If you use `gm_evar_portfolio` in your research, please consider citing us by using the following bibtex:
```
@article{luxenberg2024portfolio,
  title={Portfolio Construction with Gaussian Mixture Returns and Exponential Utility via Convex Optimization},
  author={Luxenberg, Eric and Boyd, Stephen},
  journal={Optimization and Engineering},
  volume={25},
  number={1},
  pages={555--574},
  year={2024},
  publisher={Springer}
}
```
