import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as smapi

# Read data
data_file = 'BGSgirls.csv'
data = pd.read_csv(data_file)

# Select columns of interest
columns_of_interest = ['HT2', 'HT9', 'WT2', 'WT9', 'ST9', 'BMI18']
subset_data = data[columns_of_interest]

sns.pairplot(subset_data)
plt.savefig('/home/darbvt/ds5020_hw11/3.3scatterplot.png')

print ("This is the correlation matrix")
print (subset_data.corr())

# Plot marginal plots of BMI18 versus WT9 and ST9
sns.pairplot(subset_data, x_vars=['WT9', 'ST9'], y_vars=['BMI18'], height=5)
plt.savefig('/home/darbvt/ds5020_hw11/marginal_plots_BMI18_vs_WT9_and_ST9.png')


# Plot the plot of ST9 versus WT9
sns.pairplot(subset_data, x_vars=['WT9'], y_vars=['ST9'], height=5)
plt.savefig('/home/darbvt/ds5020_hw11/ST9_vs_WT9_plot.png')

# Fit linear regression model BMI18 ~ WT9
X_wt9 = subset_data[['WT9']]
X_wt9 = smapi.add_constant(X_wt9)  # Add constant term for intercept
y_bmi18 = subset_data['BMI18']
model_wt9 = smapi.OLS(y_bmi18, X_wt9)
result_wt9 = model_wt9.fit()

# Residuals of BMI18 ~ WT9
residuals_bmi18_wt9 = result_wt9.resid

# Fit linear regression model WT9 ~ ST9
X_st9 = subset_data[['ST9']]
X_st9 = smapi.add_constant(X_st9)  # Add constant term for intercept
y_wt9 = subset_data['WT9']
model_st9 = smapi.OLS(y_wt9, X_st9)
result_st9 = model_st9.fit()

# X matrix, y = target variable
X = subset_data[['HT2', 'WT2', 'HT9', 'WT9', 'ST9']]
X = smapi.add_constant(X)  # Add constant term for intercept
y = subset_data['BMI18']

# Fit the multiple linear regression model
model = smapi.OLS(y, X)
result = model.fit()



# Get the summary of the regression results
summary = result.summary()
print('This is the mlr model with Mean function given in 3.3.3')
print(summary)

# Get the residual standard error (sigma hat)
sigma_hat = result.mse_resid ** 0.5
print("Residual standard error (sigma hat):", sigma_hat)

# Get R-squared
R_squared = result.rsquared
print("R-squared:", R_squared)

# Get t-statistics for each beta_j
t_statistics = result.tvalues
print("T-statistics for each beta_j:")
print(t_statistics)

