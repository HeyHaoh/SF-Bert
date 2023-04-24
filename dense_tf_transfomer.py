from __future__ import print_function
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

def scaled_dot_product_attention(q, k, v, mask):
  
  start = time.time()
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  # 缩放 matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
  # 将 mask 加入到缩放的张量上。
  if mask is not None:
    scaled_attention_logits += ((1 - mask) * -1e9)  
  # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
  # 相加等于1。
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
  
  end = time.time()
  time_len = (end - start)*1000
#   print("{} ms".format(time_len))
  return matmul_qk, output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """分拆最后一个维度到 (num_heads, depth).
    转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    # q = tf.linalg.band_part(tf.ones([32, 320, 512]), -1, 0)
    # k = tf.linalg.band_part(tf.ones([32, 320, 512]), 0, -1)
    # v = tf.linalg.band_part(tf.ones([32, 320, 512]), -1, 0)
    # v = tf.ones([32, 320, 512])

    # np_q = np.array(q)
    # np.save('/home/songshuhui/Documents/compare_precison/q.npy',np_q)

    # np_k = np.array(k)
    # np.save('/home/songshuhui/Documents/compare_precison/k.npy',np_k)

    
    start = time.time()
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

    matout, scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
    end = time.time()
    print(end - start)
    return matout, output, attention_weights

def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    if attn_mode == 'all':
        b = tf.linalg.band_part(tf.ones([n, n]), -1, 0)
    elif attn_mode == 'local':
        bandwidth = local_attn_ctx
        ctx = tf.minimum(n - 1, bandwidth - 1)
        b = tf.linalg.band_part(tf.ones([n, n]), ctx, 0)
    elif attn_mode == 'strided':
        stride = local_attn_ctx
        x = tf.reshape(tf.range(n, dtype=tf.int32), [n, 1])
        y = tf.transpose(x)
        z = tf.zeros([n, n], dtype=tf.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = tf.equal(tf.math.floormod(q - k, stride), 0)
        c3 = tf.logical_and(c1, c2)
        b = tf.cast(c3, tf.float32)
    else:
        raise ValueError('Not yet implemented')
    b = tf.reshape(b, [1, 1, n, n])
    return b

if __name__ == '__main__':
    temp_mha = MultiHeadAttention(d_model=512, num_heads=4)
    y = tf.random.uniform((32, 320, 512))  # (batch_size, encoder_sequence, d_model)
    # y = tf.ones([32, 320, 512])
    # np_y = np.array(y)
    # np.save('/home/songshuhui/Documents/compare_precison/input.npy',np_y)
    full_mask = tf.cast(get_attn_mask(320, attn_mode='all'), tf.float32)
    for i in range(1000):
        matout, out, attn= temp_mha(y, k=y, q=y, mask=full_mask)
    # print(y)
    # print(matout[0, 0, :, :])
    # np_matout = np.array(matout)
    # np.save('/home/songshuhui/Documents/compare_precison/matout.npy', np_matout)
    # np_attn = np.array(attn)
    # np.save('/home/songshuhui/Documents/compare_precison/attn.npy', np_attn)
    # np_out = np.array(out)
    # np.save('/home/songshuhui/Documents/compare_precison/out.npy', np_out)
    # print(attn[0,0,:,:])