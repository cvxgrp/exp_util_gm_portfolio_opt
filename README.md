# gm_evar_portfolio
Minimal entropic value at risk (EVaR) portfolio construction under a Gaussian mixture model of returns in Python 

Covariance prediction via convex optimization (`covpred`) in Python.
This repository implements the EVaR objective in graph form described in the paper [Portfolio construction with Gaussian mixture returns and exponential utility via convex optimization](https://web.stanford.edu/~boyd/papers/exp_util_gm_ret_portfolio.html).


## Optomizing a portfolio

With ```mus```, ```sigmas```, and ```pi``` lists of the respective mixture component means, covariance matrices, and mixture weights of your Gaussian mixture return model, ```L``` the leverage limit, and ```alpha``` the EVaR level, simply run
```w,delta,evar = min_EVaR_portfolio(alpha,L,mus,sigmas,pi)``` . 

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
