# ML-GOOD
This is the official pytorch implementation of the paper "ML-GOOD: Towards Multi-Label Graph Out-Of-Distribution Detection".



## Multi-Label Datasets (see `data_loader.py`)

|Datasets     |\# Nodes | \# Edges | \# Graphs | \# Labels |
|:--------:|:-----------:|:---------:|:-----------:|:-------:|
|OGB-Proteins | 132,534          | 21,446,852        | 1          |       & 112   |
|PPI         | 44,906            | 44,906            | 20         |       & 121   |
|DBLP        | 28,706            | 68,335            | 1          |       & 4     |
|PCG         |3,233              | 74,702            | 1          |       & 15    |
|HumLoc      | 3,106             | 18,496            | 1          |      & 14     |
|EukLoc      | 7,766             | 13,818            | 1          |      & 22     |

- PPI(Protein-Protein Interaction)
In general, if two proteins are involved in a life process or perform a function together, it is considered that there is an interaction between the two proteins.
The gene ontology base acts as a label(121 in total), and labels are not one-hot encoding.
> Zitnik, M.; and Leskovec, J. 2017. Predicting multicellular function through multi-layer tissue networks. Bioinformatics, (14): i190–i198.

- DBLP
There are four types of nodes in the DBLP data set: author, paper, term, and conference.
We use its multi-label forms from [MLGNC](https://github.com/Tianqi-py/MLGNC).
> Zhao, T.; Dong, T. N.; Hanjalic, A.; and Khosla, M. 2023. Multi-label Node Classification On Graph-Structured Data. TMLR, 2023.

- PCG

- HumLoc
 
- EukLoc

- OGBN-Proteins
> Hu, W.; Fey, M.; Zitnik, M.; Dong, Y.; Ren, H.; Liu, B.; Catasta, M.; and Leskovec, J. 2020. Open Graph Benchmark: Datasets for Machine Learning on Graphs. In NeurIPS, 22118–22133.
  ## Parameter
  
|Datasets| $m_{out}$ | $m_{in}$ |
|:-------:|:--------:|:----------:|
|OGB-Proteins|  \{-45, -35, -25, -15\} | \{-75, -65, -55\} |
|PCG|  \{-8, -5, -2, 1\} | \{-12, -9, -6\} |
|DBLP| \{-5, -2, 1, 4\} | \{-9, -5, -1\}  | 
|HumLoc|  \{-12, -6, -3, 0\} | \{-20, -15, -5\}|

  ## Run
  single-dataset
  ```bash
  python main.py --method mlgood --dataset pcg --ood_type feature --use_reg --m_in -9 --m_out -5 --lamda 0.01 --use_emo
  ```
  cross-dataset
  ```bash
  python main.py --method mlgood --dataset ppipcg --use_reg --m_in -9 --m_out -4 --lamda 0.01 --use_emo
  ```
