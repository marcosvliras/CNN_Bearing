from tensorflow import keras


def generate_model(train, label_train, epochs=20):

    # model definition
    model2 = keras.Sequential()

    # layers
    model2.add(keras.layers.Conv2D(
        filters=32,
        kernel_size=5,
        padding='same',
        activation='relu',
        input_shape=train.shape[1:]))

    model2.add(keras.layers.MaxPooling2D(
        pool_size=[2, 2],
        strides=2))

    model2.add(keras.layers.Conv2D(
        filters=64,
        kernel_size=5,
        padding='same',
        activation='relu',
        input_shape=train.shape[1:]))

    model2.add(keras.layers.MaxPooling2D(
        pool_size=[2, 2],
        strides=2))

    model2.add(keras.layers.Flatten())
    model2.add(keras.layers.Dense(256, activation='relu'))
    model2.add(keras.layers.Dense(256, activation='relu'))
    model2.add(keras.layers.Dense(530, activation='relu'))
    model2.add(keras.layers.Dense(1, activation='linear'))

    # compile
    model2.compile(optimizer='adam', loss='mse', metrics='mse')

    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    epochs = epochs
    history = model2.fit(
        train, label_train, epochs=epochs, batch_size=50,
        validation_split=0.2, callbacks=[callback])

    return model2, history
