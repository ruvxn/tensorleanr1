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

#training the model

x_train = temp_df['Celsius']
y_train = temp_df['Fahrenheit']

print(x_train.shape)
print(y_train.shape)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=[1]))

print(model.summary())

#compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1.0),loss='mean_squared_error')

epochs_hist = model.fit(x_train, y_train, epochs=100)

#Evaluate Model
epochs_hist.history.keys()

plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])


model.get_weights()#this returns the value thats almost equal to (9/5)


Temp_C = np.array([-32])  # Convert to a NumPy array
Temp_F = model.predict(Temp_C)

print("Temperature in Fahrenheit using trained ANN: ", Temp_F)






