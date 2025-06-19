 To install basic/necessary libraries
pip install pandas numpy matplotlib seaborn scikit-learn
# Import necessary libraries
import pandas as pd # data manipulation
import numpy as np # numerical python - linear algebra

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# load the dataset
df = pd.read_csv('water_quality.csv', sep=';')
df.info() # dataset info

# rows and cols
df.shape
(2861, 11)
# Statistics of the data
df.describe().T

# date is in object - date format
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
df.info()

df = df.sort_values(by=['id', 'date'])
df.head()
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df.head()
df.columns
pollutants = ['O2', 'NO3', 'NO2', 'SO4',
       'PO4', 'CL']