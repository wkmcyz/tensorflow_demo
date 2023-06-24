import tensorflow as tf

if __name__ == '__main__':
    path = ""
    new_model = tf.keras.models.load_model(
        '/Users/wkm/PycharmProjects/tensorflowDemo/numerical_classification/bank_marketing/history_models/2023_06_07_19_09_24/model.h5')
    new_model.summary()
