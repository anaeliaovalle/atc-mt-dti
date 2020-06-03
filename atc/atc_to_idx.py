import numpy as np
import csv
import pdb
import pdb
import tensorflow as tf
import _pickle as cPickle



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
    try: 

        # map chembl to smiles
        seq_to_chembl_file_name = "../data/kiba/seq_to_id.cpkl"
        chembl_to_smiles = {}
        with open(seq_to_chembl_file_name, 'rb') as seq_handle:
            (mseq_to_id, _) = cPickle.load(seq_handle)

        for seq_idx, (smiles, chembl) in mseq_to_id.items():
            chembl_to_smiles[chembl] = seq_idx

        # load atc and map to chembl
        atc_file_name = 'data/ATC_embedding.pkl'
        atc_embedding = np.load(atc_file_name, allow_pickle=True)

        atc_to_chembl_file_name = "./data/ATC_Drug_to_CHEMBL.csv"
        atc_chembl = {}
        with open(atc_to_chembl_file_name, newline='\n') as csvfile:
           reader = csv.DictReader(csvfile)
           for idx, row in enumerate(reader):
               atc_chembl[row['CHEMBL']] = idx

        # map atc chembl to smiles if exists in kiba otherwise na
        atc_chembl_to_smiles = {}
        for c, s in chembl_to_smiles.items(): 
            if c in atc_chembl.keys():
                atc_chembl_to_smiles[c] = (s, atc_chembl[c])
        
        # 5 drugs that overlap between ATC chembl and chembl in kiba
        atc_chembal_overlap_list = list(atc_chembl_to_smiles.keys())
        atc_idx_overlap = np.array([val[1] for _, val in atc_chembl_to_smiles.items()])

        # adjust existing embedding to only consider the overlapping drugs
        atc_overlap_embedding = atc_embedding[atc_idx_overlap]
        
        # write out smiles vocabulary with key being cd
        with open('./data/atc_overlap_vocab.txt', 'w') as handler: 
            for chembl, val in atc_chembl_to_smiles.items():
                smiles = [int(x) for x in val[0] if x != ',']
                is_non_zero = map(lambda x: x > 0, smiles)
                embed_key = sum(smiles) + sum(is_non_zero)
                print(embed_key)
                handler.write(str(embed_key) + '\n')

        # write out new embedding
        np.save('./data/atc_overlap_embedding',atc_overlap_embedding)

        

        # convert input_id to chembl
        # atc_chembl_smiles = [chembl_to_smiles.get(atc_chembl, 0)[q][1] for q in query]

        

    except Exception as e: 
        print(e)
        pdb.set_trace()



if __name__ == "__main__": 
    main()