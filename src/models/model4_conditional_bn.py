import tensorflow as tf
from keras.src.layers import LSTM


class ConditionalBatchNorm1D(tf.keras.layers.Layer):
    def __init__(self, num_batches=3, **kwargs):
        super(ConditionalBatchNorm1D, self).__init__(**kwargs)
        self.num_batches = num_batches
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        self.gamma = self.add_weight(
            name='gamma',
            shape=(self.num_batches, feature_dim),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(self.num_batches, feature_dim),
            initializer='zeros',
            trainable=True
        )
        super(ConditionalBatchNorm1D, self).build(input_shape)

    def call(self, inputs, training=None):
        features, batch_ids = inputs
        normalized = self.bn(features, training=training)
        batch_gamma = tf.gather(self.gamma, batch_ids)
        batch_beta = tf.gather(self.beta, batch_ids)
        batch_gamma = tf.expand_dims(batch_gamma, axis=1)
        batch_beta = tf.expand_dims(batch_beta, axis=1)
        return normalized * batch_gamma + batch_beta

    def get_config(self):
        config = super(ConditionalBatchNorm1D, self).get_config()
        config.update({'num_batches': self.num_batches})
        return config


# This constructs the layers of the model and returns it.
def build_model(input_shape, output_dim=1, num_batches=3):
    # Conditional BatchNorm uses batch IDs to modulate channel statistics.
    # It replaces the BatchNormalization layer found in other models.
    seq = tf.keras.Input(shape=input_shape, name='sequence_input')
    batch = tf.keras.Input(shape=(), dtype=tf.int32, name='batch_id_input')

    x = tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu')(seq)
    x = ConditionalBatchNorm1D(num_batches=num_batches)([x, batch])
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu')(x)
    x = ConditionalBatchNorm1D(num_batches=num_batches)([x, batch])
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=False))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    output = tf.keras.layers.Dense(output_dim, activation='linear')(x)

    model = tf.keras.Model(inputs=[seq, batch], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
