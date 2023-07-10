import tensorflow as tf

# Restore the saved model
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, './model/model.ckpt')

# ... code for preparing the input data for prediction ...

# Make predictions
prediction = sess.run(model.output, feed_dict={model.input: input_data})

# Print the predictions
print(prediction)
