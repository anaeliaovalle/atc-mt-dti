import numpy as np
import csv
import pdb
import pdb
import tensorflow as tf
import _pickle as cPickle

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
        
        # write out smiles vocabulary with key being key=sum(indices) + number of nonzeroes
        with open('./data/atc_overlap_vocab.txt', 'w') as handler: 
            for chembl, val in atc_chembl_to_smiles.items():
                smiles = [int(x) for x in val[0] if x != ',']
                is_non_zero = map(lambda x: x > 0, smiles)
                embed_key = sum(smiles) + sum(is_non_zero)
                print(embed_key)
                handler.write(str(embed_key) + '\n')

        # write out new embedding
        np.save('./data/atc_overlap_embedding',atc_overlap_embedding)


    except Exception as e: 
        print(e)
        pdb.set_trace()



if __name__ == "__main__": 
    main()