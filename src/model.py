import tensorflow as tf
from keras.src.layers import LSTM
from src import config


# Custom batch correction layer to adjust for differences in sources.
class BatchCorrectionLayer(tf.keras.layers.Layer):
    
    def __init__(self, num_batches=3, **kwargs):
        super(BatchCorrectionLayer, self).__init__(**kwargs)
        self.num_batches = num_batches
    
    def build(self, input_shape):
        # input_shape[0] is features, input_shape[1] is batch_ids.
        feature_dim = input_shape[0][-1]
        
        # Learnable batch-specific scaling parameters (gamma).
        self.gamma = self.add_weight(
            name='gamma',
            shape=(self.num_batches, feature_dim),
            initializer='ones',
            trainable=True
        )
        
        # Learnable batch-specific shift parameters (beta).
        self.beta = self.add_weight(
            name='beta',
            shape=(self.num_batches, feature_dim),
            initializer='zeros',
            trainable=True
        )
        
        super(BatchCorrectionLayer, self).build(input_shape)
    
    def call(self, inputs):
        features, batch_ids = inputs
        
        # Gather batch-specific parameters for each sample.
        batch_gamma = tf.gather(self.gamma, batch_ids)
        batch_beta = tf.gather(self.beta, batch_ids)
        
        # Batch-specific transformation: output = gamma * features + beta.
        corrected = batch_gamma * features + batch_beta
        
        return corrected
    
    def get_config(self):
        config = super(BatchCorrectionLayer, self).get_config()
        config.update({'num_batches': self.num_batches})
        return config


# This constructs the layers of the model and returns it.
def build_model(input_shape, output_dim=1, num_batches=3):
    if config.USE_BATCH_CORRECTION:
        # Requires two inputs: sequences and batch IDs.
        sequence_input = tf.keras.Input(shape=input_shape, name='sequence_input')
        batch_id_input = tf.keras.Input(shape=(), dtype=tf.int32, name='batch_id_input')
        
        x = tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu')(sequence_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        
        x = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        
        x = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=False))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Apply batch correction before final dense layers.
        x = BatchCorrectionLayer(num_batches=num_batches)([x, batch_id_input])
        
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        output = tf.keras.layers.Dense(output_dim, activation='linear')(x)
        
        model = tf.keras.Model(inputs=[sequence_input, batch_id_input], outputs=output)
        
    else:
        # Original model without batch correction.
        model = tf.keras.Sequential([
            tf.keras.Input(shape=input_shape),

            tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),

            tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),

            tf.keras.layers.Bidirectional(LSTM(64, return_sequences=False)),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(output_dim, activation='linear')
        ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model