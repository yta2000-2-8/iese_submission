import tensorflow as tf

# Create a model using high-level tf.keras.* APIs
model = tf.keras.models.load_model('./saved_models/ECG_net_tf.h5')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()

# Save the model.
with open('saved_models/af_detect.tflite', 'wb') as f:
  f.write(tflite_model)
