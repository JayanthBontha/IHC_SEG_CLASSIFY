import pandas as pd
from PIL import Image
df =pd.read_csv("./finaldata.csv")

train,test=[],[]
for index, row in df.iterrows():
    train.append([row['R'],row['G'],row['B']])
    if row['Prediction'] == 'Brown':
        test.append(1)
    if row['Prediction'] == 'Blue':
        test.append(0)


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(train, test, test_size=0.2, random_state=42)


import tensorflow as tf
from tensorflow.contrib import keras
import numpy as np
from sklearn.model_selection import train_test_split

print(tf.__version__)

model = keras.models.Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=(3,)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# Convert data to NumPy arrays
train_input_array = train_x
train_labels = train_y
test_input_array = test_x
test_labels = test_y
# Normalize pixel values between 0 and 1
train_input_array = np.array(train_input_array)
test_input_array = np.array(test_input_array)
train_input_array = train_input_array / 255.0
test_input_array = test_input_array / 255.0
train_input_array = np.reshape(train_input_array, (1759, 3))
test_input_array = np.reshape(test_input_array, (440, 3))

test_labels = np.array(test_labels)




history = model.fit(train_input_array, train_labels, epochs=10)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_input_array, test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
sess = tf.Session()

sess = tf.Session()
# Initialize variables
sess.run(tf.global_variables_initializer())
# Save the model
predictions = sess.run(model.output, feed_dict={model.input: test_input_array})

print(predictions)
saver = tf.train.Saver()
saver.save(sess, './model/model.ckpt')


# import tensorflow as tf
# # Create a session and initialize variables
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()

# # Restore the saved variables
# saver.restore(sess, './model/model.ckpt')

# # Make predictions
# predictions = sess.run(model.output, feed_dict={model.input: test_input_array})

# print(predictions)




