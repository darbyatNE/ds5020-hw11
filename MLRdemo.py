import numpy as np
import pandas as pd
import seaborn as sb
import statsmodels as smapi
import matplotlib.pyplot as plt

# to load csv as pandas data frame
# dat = pd.read_csv(...)

# I will simulate data instead

# sample size
n = 100

# true parameter vector
vbeta = np.array([2,-.5,1.5,.5])
p = len(vbeta)-1

# simulated x-values with intercept column
X = np.random.uniform(-1,1,[n,p])
X = np.column_stack((np.ones(n),X))

# generate error terms
sig = 1
veps = np.random.normal(0,sig,n)

# generate response vector
y = np.matmul(X,vbeta)+veps

# scatterplot matrix of X matrix
dat = pd.DataFrame(np.column_stack((y,X[:,1:])))
sb.pairplot(dat)

# correlation matrix
# (note that np.corrcoef is expecting
# a matrix that is (# vars) x (# obs), i.e., p x n
# whereas we typically have n x p, thus the transpose
np.round(np.corrcoef(dat.T),3)

# fit a linear regression model
m = smapi.OLS(y,X,hasconst=True)
m.fit()
m.summary()


