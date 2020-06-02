import numpy as np
import pdb
import tensorflow as tf

EMB_DIM = 128
def load_pretrained_atc():
    return ["a", "cat", "sat", "on", "the", "mat"], np.random.rand(6, EMB_DIM)

def get_train_atc():
    return ["a", "dog", "sat", "on", "the", "mat"]

def embed_tensor(string_tensor, trainable=True):
  """
  Convert List of strings into list of indices then into 128d vectors
  """
  # ordered lists of vocab and corresponding (by index) 300d vector
  pretrained_vocab, pretrained_embs = load_pretrained_atc()
  train_vocab = get_train_atc()
  only_in_train = list(set(train_vocab) - set(pretrained_vocab))
  vocab = pretrained_vocab + only_in_train

  # Set up tensorflow look up from string word to unique integer
  vocab_lookup = tf.contrib.lookup.index_table_from_tensor(
    mapping=tf.constant(vocab),
    default_value=len(vocab))
  string_tensor = vocab_lookup.lookup(string_tensor)

  # define the word embedding
  pretrained_embs = tf.get_variable(
      name="embs_pretrained",
      initializer=tf.constant_initializer(np.asarray(pretrained_embs), dtype=tf.float32),
      shape=pretrained_embs.shape,
      trainable=trainable)
  train_embeddings = tf.get_variable(
      name="embs_only_in_train",
      shape=[len(only_in_train), EMB_DIM],
      initializer=tf.random_uniform_initializer(-0.04, 0.04),
      trainable=trainable)
  unk_embedding = tf.get_variable(
      name="unk_embedding",
      shape=[1, EMB_DIM],
      initializer=tf.random_uniform_initializer(-0.04, 0.04),
      trainable=False)

  embeddings = tf.concat([pretrained_embs, train_embeddings, unk_embedding], axis=0)

  return tf.nn.embedding_lookup(embeddings, string_tensor)

def main():
    file = 'data/ATC_embedding.pkl'
    try: 
        embed = np.load(file, allow_pickle=True)
        pdb.set_trace()

        

    except Exception as e: 
        print(e)
        pdb.set_trace()



if __name__ == "__main__": 
    main()