# exp_util_gm_portfolio_opt
Exponential utility and entropic value at risk (EVaR) portfolio construction under a Gaussian mixture model of returns in Python. 
This repository implements the exponential utility and EVaR objective portfolio optimization problems described in the paper [Portfolio construction with Gaussian mixture returns and exponential utility via convex optimization](https://web.stanford.edu/~boyd/papers/exp_util_gm_ret_portfolio.html).


## Optimizing a portfolio
To run a minimal example, simply load example problem data defined in problem_data.py via

```from problem_data import mus, Sigmas, pi, n, k```.

This defines ```mus```, ```Sigmas```, and ```pi``` as lists of the respective mixture component means, covariance matrices, and mixture weights of a Gaussian mixture return model with 4 components, for an 11 dimensional return distribution. 

After loading the EVaR portfolio optimization code via 

```from gm_evar_portfolio import min_EVaR_portfolio```

and with ```L``` the leverage limit and ```alpha``` the EVaR level, simply run

```w,delta,evar = min_EVaR_portfolio(alpha,L,mus,Sigmas,pi)``` 

to generate a minimal EVaR portfolio. 

## Citing
If you use `gm_evar_portfolio` in your research, please consider citing us by using the following bibtex:
```
@misc{luxenberg2022evar,
  title={Portfolio construction with Gaussian mixture returns and exponential utility via convex optimization},
  author={Luxenberg, Eric and Boyd, Stephen}
  year={2022},
  howpublished={\texttt{https://web.stanford.edu/~boyd/papers/exp_util_gm_ret_portfolio.html}}
}
```
