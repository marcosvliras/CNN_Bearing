from tensorflow import keras, random
import pickle
import os

def check_if_model_exists(bearing, epochs):
    model = os.path.isfile(f"model/{epochs}model{bearing}.pkl")
    history = os.path.isfile(f"model/{epochs}history{bearing}.pkl")
    if model and history:
        return True
    else:
        return False

def generate_model(train, label_train, bearing, epochs=20):

    if check_if_model_exists(bearing, epochs):
        with open(f"model/{epochs}model{bearing}.pkl", 'rb') as f:
            model2 = pickle.load(f)
        with open(f"model/{epochs}history{bearing}.pkl", 'rb') as f:
            history = pickle.load(f)
        return model2, history

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

    random.set_seed(42)
    # compile
    model2.compile(optimizer='adam', loss='mse', metrics='mse')

    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model2.fit(
        train, label_train, epochs=epochs, batch_size=50,
        validation_split=0.2, callbacks=[callback])

    os.makedirs('model', exist_ok=True)

    with open(f"model/{epochs}model{bearing}.pkl", 'wb') as f:
        pickle.dump(model2, f)

    with open(f"model/{epochs}history{bearing}.pkl", 'wb') as f:
        pickle.dump(history, f)

    return model2, history
