import tensorflow as tf
from keras.src.layers import LSTM


class ConditionalFiLMLayer(tf.keras.layers.Layer):
    def __init__(self, num_batches=3, hidden_units=64, **kwargs):
        super(ConditionalFiLMLayer, self).__init__(**kwargs)
        self.num_batches = num_batches
        self.hidden_units = hidden_units
        self.embed = tf.keras.layers.Embedding(num_batches, hidden_units)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_units, activation='relu'),
            tf.keras.layers.Dense(hidden_units * 2, activation='linear')
        ])

    def call(self, inputs):
        features, batch_ids = inputs
        batch_embed = self.embed(batch_ids)
        params = self.mlp(batch_embed)
        gamma, beta = tf.split(params, num_or_size_splits=2, axis=-1)
        return features * gamma + beta

    def get_config(self):
        config = super(ConditionalFiLMLayer, self).get_config()
        config.update({
            'num_batches': self.num_batches,
            'hidden_units': self.hidden_units
        })
        return config


# This constructs the layers of the model and returns it.
def build_model(input_shape, output_dim=1, num_batches=3):
    # Conditional FiLM uses an MLP on batch embedding to generate gamma/beta.
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

    # Batch Correction Layer
    x = ConditionalFiLMLayer(num_batches=num_batches, hidden_units=128)([x, batch])

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    output = tf.keras.layers.Dense(output_dim, activation='linear')(x)

    model = tf.keras.Model(inputs=[seq, batch], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
