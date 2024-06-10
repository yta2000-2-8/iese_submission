# import tensorflow as tf
# from tensorflow.keras import layers, models
#
# def AFNet():
#     model = models.Sequential([
#         layers.Conv2D(filters=3, kernel_size=(6, 1), strides=(2, 1), padding='valid', activation='relu'),
#         # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
#
#         layers.Conv2D(filters=5, kernel_size=(5, 1), strides=(2, 1), padding='valid', activation='relu'),
#         # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
#
#         layers.Conv2D(filters=10, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
#         # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
#
#         layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
#         # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
#
#         layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
#         # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
#
#         layers.Flatten(),  # 将卷积层输出扁平化处理，以便输入到全连接层
#         # layers.Dropout(0.5),
#         layers.Reshape((-1, 20)),
#         # RNN
#         layers.LSTM(128, return_sequences=True),
#         layers.LSTM(64),
#         # 全连接层
#         layers.Dense(10, activation='relu'),
#         layers.Dense(2)
#     ])
#
#     return model

# input=tf.random.normal([64,1250,1,1])  #32,1250可以
# net=AFNet()
# y=net(input)
# print("output=",y.shape)

import tensorflow as tf
from tensorflow.keras import layers, models

def AFNet():
  model = models.Sequential([
      layers.Conv2D(filters=3, kernel_size=(6, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=5, kernel_size=(5, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=10, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

      layers.Flatten(),  # 将卷积层输出扁平化处理，以便输入到全连接层
      # layers.Dropout(0.5),
      layers.Dense(10, activation='relu'),
      layers.Dense(2)
  ])

  return model
