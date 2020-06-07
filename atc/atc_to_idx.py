import numpy as np
import csv
import pdb
import pdb
import tensorflow as tf
import _pickle as cPickle

def main():
    #step 1: kiba encoded smiles -> chembl
    #step 2: atc drugs -> chembl
    #step 3: for drug in atc_chembl, if it exists in kiba chembl, grab those smiles
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

        # map atc chembl to kiba smiles if exists in kiba otherwise na
        atc_chembl_to_smiles = {}
        for c, s in chembl_to_smiles.items(): 
            if c in atc_chembl.keys():
                atc_chembl_to_smiles[c] = (s, atc_chembl[c])
        
        # 5 drugs that overlap between ATC chembl and chembl in kiba
        atc_chembal_overlap_list = list(atc_chembl_to_smiles.keys())
        atc_idx_overlap = np.array([val[1] for _, val in atc_chembl_to_smiles.items()])

        # adjust existing embedding to only consider the overlapping drugs
        atc_overlap_embedding = atc_embedding[atc_idx_overlap]
        
        # write out smiles vocabulary with key being key=sum(indices) + number of nonzeroes
        with open('./data/atc_overlap_vocab.txt', 'w') as handler: 
            handler.write('EMBED_ID,ROW_IDX\n')
            for idx, (chembl, val) in enumerate(atc_chembl_to_smiles.items()):
                str_smiles = val[0]
                smiles = [int(x) for x in str_smiles if x != ',']
                is_non_zero = map(lambda x: x > 0, smiles)
                embed_key = sum(smiles) + sum(is_non_zero)
                # print(embed_key)
                handler.write(f'{embed_key}, {idx}\n')

        # write out new embedding
        assert len(atc_overlap_embedding) == len(atc_chembal_overlap_list), "overlap embedding not equal to overlap list"
        np.save('./data/atc_overlap_embedding', atc_overlap_embedding)


    except Exception as e: 
        print(e)
        pdb.set_trace()



if __name__ == "__main__": 
    main()
    
    import tensorflow as tf

    keys = []
    values = []
    with open('./data/atc_overlap_vocab.txt', newline='\n') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # pdb.set_trace()
            keys.append(row['EMBED_ID'])
            values.append(row['ROW_IDX'])
    
    keys = tf.constant(keys)
    values = tf.constant(values)

    import pdb
    pdb.set_trace()

    # with open('./data/atc_overlap_vocab.txt', 'r') as handler:
    #     import pdb
    #     pdb.set_trace()
    #     atc_vocab = handler.read()
    #     atc_vocab = atc_vocab.split('\n')[:-1] #TODO    
    # step1: map from 111 -> 0 
    # step2: map 0 to embedding


    # keys_tensor = tf.constant([111, 222])
    # vals_tensor = tf.constant([[1,2], [3,4]])
    # input_tensor = tf.constant([111, 5])
    # table = tf.contrib.lookup.HashTable(
    #     tf.contrib.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)

    # import pdb
    # pdb.set_trace()

    # print(table.lookup(input_tensor))