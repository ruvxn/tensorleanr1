import tensorflow as tf
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
temp_df = pd.read_csv('dataset/Celsius+to+Fahrenheit.csv')

#learn aboout dataset
print(temp_df.head(5))
print(temp_df.describe())

sns.scatterplot(x='Celsius', y='Fahrenheit', data=temp_df)

plt.xlabel("Celsius")
plt.ylabel("Fahrenheit")
plt.title("Celsius to Fahrenheit Conversion")

plt.show()

