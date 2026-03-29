import tensorflow as tf
from tensorflow.keras import layers, regularizers

def build_lstm_model(
    input_shape=(30, 3),
    lstm_units=[128, 64],
    dropout_rate=0.3,
    l2_lambda=1e-4,
):
    inputs = tf.keras.Input(shape=input_shape, name="vitals_sequence")
    x = inputs

    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)
        x = layers.LSTM(
            units,
            return_sequences=return_seq,
            kernel_regularizer=regularizers.l2(l2_lambda),
            recurrent_regularizer=regularizers.l2(l2_lambda / 2),
            name=f"lstm_{i+1}"
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

    x = layers.Dense(32, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    output = layers.Dense(1, activation="sigmoid", name="hypoxia_probability")(x)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="HypoxiaLSTM")
    return model

if __name__ == "__main__":
    model = build_lstm_model()
    model.summary()
