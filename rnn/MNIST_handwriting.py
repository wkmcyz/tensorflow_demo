from keras import Sequential
from keras.datasets import mnist
from keras.layers import LSTM, Dropout, Dense
from keras.utils import to_categorical

if __name__ == '__main__':
    # load data from mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train / 255
    X_test = X_test / 255
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    model = Sequential()
    model.add(
        LSTM(
            units=28,
            input_shape=(X_train.shape[1:]),
            activation="relu",
            return_sequences=True
        )
    )
    model.add(LSTM(28, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="softmax"))

    # for l in model.layers:
    #     print(l.input_shape, l.output_shape, l.count_params())

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(X_train, Y_train, validation_split=0.2, epochs=10, batch_size=128, verbose=2)

    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"loss,{loss}; accuracy : {accuracy}")
