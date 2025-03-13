# Violin: a Volumetric Injection Attack Against SSE

### Implementations of Violin

This repository contains Python implementation of the attacks presented in:

Violin: Powerful Volumetric Injection Attack Against Searchable Encryption with Optimal Injection Size  



### Install Required Packages

Before running, you need to install the necessary packages

```shell
pip3 install -r requirements.txt
```



### Summary of files

`selectivity_*.py`: evaluates the impact of query selectivity on the attack.

`m_enron.py`: evaluates the impact of the number of queries, including recovery rate, number of injection files, and injection volume.

`n_*.py`: evaluates the impact of the number of keywords in the set on the attack, including recovery rate, injection volume, and attack time.

`padding_*_*.py`: evaluates the impact of padding (static padding with SEAL and dynamic padding with ShieldDB) on the attack.

`ThresholdCounter*.py`: evaluates the injection volume for attacks with threshold countermeasures.

`Update.py`: evaluates the recovery rate of attacks when the client actively updates.

`injectionPadding.py`: evaluates the impact of different injection padding parameters on the recovery rate.

### Datasets

The download links for the datasets are as follows:

- Enron: https://www.cs.cmu.edu/~enron/
- Lucene: http://mail-archives.apache.org/mod_mbox/lucene-java-user
- Movies: http://www.cs.cmu.edu/~ark/personas/

We have processed these datasets (tokenization, filtering, etc.), and the processed pickle files are located in the `Datasets` directory.

- `doc_size_*.pkl`: contains the size of each file in the dataset.
- `*_vol_access.pkl`: a dictionary that shows the relationship between keywords and the sizes of the files containing them.

### Citations
> @ARTICLE{10891733,
  author={Zhang, Lei and Wang, Jianfeng and Wu, Jiaojiao and Wang, Yunling and Sun, Shi-Feng},
  journal={IEEE Transactions on Dependable and Secure Computing}, 
  title={Violin: Powerful Volumetric Injection Attack Against Searchable Encryption With Optimal Injection Size}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TDSC.2025.3543248}}






