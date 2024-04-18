import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('water.csv')

# Extract the columns of interest
regressors = ['OPBPC', 'OPRC', 'OPSLAKE']
response = 'BSAAM'

# Create scatterplot matrix
scatterplot_matrix = pd.plotting.scatter_matrix(data[regressors + [response]], figsize=(10, 8), diagonal='hist')

# Add titles to the diagonal subplots
for ax in scatterplot_matrix.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize=10, rotation=90)
    ax.set_ylabel(ax.get_ylabel(), fontsize=10, rotation=0, ha='right')



plt.tight_layout()
plt.savefig('/home/darbvt/ds5020_hw11/3.6scatterplot.png')

# Calculate the correlation matrix
correlation_matrix = data[regressors + [response]].corr()

# Display the correlation matrix
print(correlation_matrix)

# You can also visualize the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('/home/darbvt/ds5020_hw11/3.6corrmatrix.png')

import pandas as pd
import statsmodels.api as sm

# Load the data
data = pd.read_csv('water.csv')

# Define the regressors and the response variable
regressors = ['OPBPC', 'OPRC', 'OPSLAKE']
response = 'BSAAM'

# Add a constant term to the regressors
X = sm.add_constant(data[regressors])

# Fit the regression model
model = sm.OLS(data[response], X).fit()

# Print the regression summary
print(model.summary())
