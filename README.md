# **BTC**
BTC is a transformer-based label prediction model. It accepts gene profile and TCR β-chain CDR3 sequence information from the same T cell. We use a transformer to learn a contextual representation for each residue by attending to its position and the surrounding amino acids within the CDR3 sequence. This residue-level representation is then aggregated by an MLP to derive a sequence-level representation of the TCR. Additionally, we employ the VAE-basic model to reduce the dimensionality of the high-dimensional gene expression data. 

![图片1](https://github.com/user-attachments/assets/9f96dc10-818e-4221-a941-4a670ad8d558)

## Preprocess
Pairing scRNA-seq and scTCR-seq data is required, including gene expression profiles and TCR annotations. For instance, the file XXX_filtered_contig_annotations.csv. The preprocessing of the scTCR-seq data is as follows:
![1736437749014](https://github.com/user-attachments/assets/29c5dba5-8265-4f1a-ad80-486a2b42a66f)

## Basic Usage
1. Download the preprocessing script from the repository.
```R
Rscript preprocess.R ./data
```
2. Prepare the input files in the required format.
3. Execute the script using the command:
```python
python merge_predict.py ./config.yaml ./data/predict_data.h5ad -o ./data/output.h5ad
```
