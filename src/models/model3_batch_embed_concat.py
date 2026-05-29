import tensorflow as tf
from keras.src.layers import LSTM


# This constructs the layers of the model and returns it.
def build_model(input_shape, output_dim=1, num_batches=3):
    # Batch embedding concatenation at the latent feature level.
    seq = tf.keras.Input(shape=input_shape, name='sequence_input')
    batch = tf.keras.Input(shape=(), dtype=tf.int32, name='batch_id_input')

    x = tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu')(seq)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=False))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Apply batch embedding before the final layers.
    batch_embed = tf.keras.layers.Embedding(num_batches, 128)(batch)
    batch_embed = tf.keras.layers.Flatten()(batch_embed)
    x = tf.keras.layers.Concatenate()([x, batch_embed])

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    output = tf.keras.layers.Dense(output_dim, activation='linear')(x)

    model = tf.keras.Model(inputs=[seq, batch], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
