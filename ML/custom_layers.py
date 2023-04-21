import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, LayerNormalization, Permute, Embedding, MultiHeadAttention

class Transpose(keras.layers.Layer):
    def __init__(self, dims=None, **kwargs):
        super().__init__(**kwargs)
        self.dims = dims

    def call(self, inputs):
        if self.dims:
            # If the user specified the dimensions to permute, transpose the input accordingly
            return Permute(self.dims)(inputs)
        else:
            # If the user did not specify any dimensions, transpose the input using the default order
            return Permute((-1, -2))(inputs)

    def compute_output_shape(self, input_shape):
        if self.dims:
            # If the user specified the dimensions to permute, compute the output shape accordingly
            return tuple([input_shape[d] for d in self.dims])
        else:
            # If the user did not specify any dimensions, compute the output shape using the default order
            return tuple(reversed(input_shape))
        

class DenseNormDrop(keras.layers.Layer):
    def __init__(self, units, dr=.05):
        super(DenseNormDrop, self).__init__()
        self.units = units
        self.dr    = dr
        self.dense = Dense(units=units, activation='relu')
        self.norm  = LayerNormalization()
        self.drop  = Dropout(dr)
        
    def call(self, inputs):
        dense_out = self.dense(inputs)
        norm_out  = self.norm(dense_out)
        drop_out  = self.drop(norm_out)
        return drop_out
    

class MaskedPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(MaskedPositionEmbedding, self).__init__()
        self.pos_emb = Embedding(input_dim=maxlen+1,
                                 output_dim=embed_dim,
                                 mask_zero=True)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        
    def compute_output_shape(self, input_shape):
        return input_shape + (embed_dim,)
    
    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=1, limit=maxlen+1, delta=1)
        positions = positions * tf.cast(tf.sign(tf.math.count_nonzero(x,axis=2)),tf.int32)
        positions = self.pos_emb(positions)
        return x + positions
    

class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads,
                                                   key_dim=embed_dim)
        self.ffn = keras.Sequential(
                                    [Dense(ff_dim, activation="gelu"),
                                     Dense(embed_dim),])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    

class DoubleTransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(DoubleTransformerBlock, self).__init__()
        self.att1 = MultiHeadAttention(num_heads=num_heads,
                                       key_dim=embed_dim)
        self.att2 = MultiHeadAttention(num_heads=num_heads,
                                       key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [Dense(ff_dim, activation="gelu"),
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)
        
    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.
        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, inputs, training):
        input_shape = tf.shape(inputs[0])
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        attn_output1 = self.att1(inputs[0], inputs[0])
        attn_output1 = self.dropout1(attn_output1, training=training)
        out1 = self.layernorm1(inputs[0] + attn_output1)
        attn_output2 = self.att2(out1, inputs[1])
        attn_output2 = self.dropout2(attn_output2, training=training)
        out2 = self.layernorm1(out1 + attn_output2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm2(out2 + ffn_output)