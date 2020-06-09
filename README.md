# Augmenting MT-DTI model with ATC drug classification for Drug-Drug Chemical Subgroup Similarity
https://www.who.int/medicines/regulation/medicines-safety/toolkit_atc/en/






### CS247 Documentation for running the ATC embedding

1) Download [data.tar.gz](https://drive.google.com/file/d/16dTynXCKPPdvQq4BiXBdQwNuxilJbozR/view?usp=sharing) to get the Kiba tfrecords
2) Run the model finetuning on the ATC Embedding

```
cd src/finetune
export PYTHONPATH='../../'
python finetune_demo.py  --use-atc --save-model-dir ./yes-atc 
```
2) Run the predictions

For running predictions on ATC embedding with Kiba data:
```
cd src/predict
export PYTHONPATH='../../'
python predict_demo.py  --use-atc --load-model-dir ../finetune/yes-atc 
```

For running predictions on ATC embedding with COVID19 data:
```
cd src/predict
export PYTHONPATH='../../'
python predict_demo.py  --covid --use-atc --load-model-dir ../finetune/yes-atc 
```


### Description of jupyter notebooks
1) src/data\_processing\_notebooks/'ATC Adjacency Matrix.ipynb'

```
This notebook was used to prepare the ATC adjacency matrix for the ATC embedding. A database of drugs with ATC codes were used to create an adjacency matrix based on the level 2 ATC classification. 
```
2) src/data\_processing\_notebooks/'Data Assembly for COVID.ipynb'
```
This notebook was used to prepare the appropriate files for running the ATC-MT-DTI model for COVID-19. The files generated are the protein.txt and ligands_can.txt files which are data files for the protein targets and drug compounds repectively.
```
3) src/data\_processing\_notebooks/'Mapping ATC drug names to ChEMBL ID.ipynb'
```
This notebook was used to map the drug names from the ATC database (1st jupyter notebook) to a common naming schema using ChEMBL IDs. This mapping was used to look up ATC embeddings for drugs with known ChEMBL IDs.
```
4) src/data\_processing\_notebooks/'ATC Adjacency Matrix.ipynb'

```
This notebook was used to generate the r-square plot for the KIBA benchmarking dataset and calculates R^2 statistic and mean-square error. An r-squared plot is generated for both the ATC-MT-DTI model and the baseline model. 
```
5) data/covid/'COVID DTI Prediction Results.ipynb'
```
This notebook was used to analyze the resulting drug target interactions for the COVID-19 dataset. This notebook reads in the result file and looks up the ChEMBL IDs for each drug to get its preferred drug name as well as its ATC classification if available.
```



### Forked Repo Documentation
-----------
# MT-DTI
An official Molecule Transformer Drug Target Interaction (MT-DTI) model

* **Author**: [Bonggun Shin](mailto:bonggun.shin@deargen.me)
* **Paper**: Shin, B., Park, S., Kang, K. & Ho, J.C.. (2019). [Self-Attention Based Molecule Representation for Predicting Drug-Target Interaction](http://proceedings.mlr.press/v106/shin19a/shin19a.pdf). Proceedings of the 4th Machine Learning for Healthcare Conference, in PMLR 106:230-248

## Required Files

* Download [data.tar.gz](https://drive.google.com/file/d/16dTynXCKPPdvQq4BiXBdQwNuxilJbozR/view?usp=sharing)
	* This includes;
		* Orginal KIBA dataset from [DeepDTA](https://github.com/hkmztrk/DeepDTA)
		* tfrecord for KIBA dataset
		* Pretrained weights of the molecule transformer
		* Finetuned weights of the MT-DTI model for KIBA fold0
* Unzip it (folder name is **data**) and place under the project root

```
cd mtdti_demo
# place the downloaded file (data.tar.gz) at "mtdti_demo"
tar xzfv data.tar.gz
```

* These files sholud be in the right places

```
mtdti_demo/data/chembl_to_cids.txt
mtdti_demo/data/CID_CHEMBL.tsv
mtdti_demo/data/kiba/*
mtdti_demo/data/kiba/folds/*
mtdti_demo/data/kiba/mbert_cnn_v1_lr0.0001_k12_k12_k12_fold0/*
mtdti_demo/data/kiba/tfrecord/*.tfrecord
mtdti_demo/data/pretrain/*
mtdti_demo/data/pretrain/mbert_6500k/*
```



## VirtualEnv

* install mkvirtualenv
* create a dti env with the following commands

```
mkvirtualenv --python=`which python3` dti
pip install tensorflow-gpu==1.12.0
```


## Preprocessing

* If downloaded [data.tar.gz](https://drive.google.com/file/d/16dTynXCKPPdvQq4BiXBdQwNuxilJbozR/view?usp=sharing), then you can skip these preprocessings


* Transform kiba dataset into one pickle file

```
python kiba_to_pkl.py 

# Resulted files
mtdti_demo/data/kiba/kiba_b.cpkl
```



* Prepare Tensorflow Record files

```
cd src/preprocess
export PYTHONPATH='../../'
python tfrecord_writer.py 

# Resulted files
mtdti_demo/data/kiba/tfrecord/*.tfrecord
```

## FineTuning

* If downloaded [data.tar.gz](https://drive.google.com/file/d/16dTynXCKPPdvQq4BiXBdQwNuxilJbozR/view?usp=sharing), then you can skip this finetuning

```
cd src/finetune
export PYTHONPATH='../../'
python finetune_demo.py 

```


## Prediction

```
cd src/predict
export PYTHONPATH='../../'
python predict_demo.py 
```



