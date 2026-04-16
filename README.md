<div align="center">

<h1>LISRec: Modeling User Preferences with Learned Item Shortcuts for Sequential Recommendation
</h1>

<h5 align="center">
<a href='https://arxiv.org/abs/2505.22130'><img src='https://img.shields.io/badge/Paper-LISRec-red?logo=arxiv&logoColor=white'></a>
<a href='https://huggingface.co/xhd0728/LISRec-MFilter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LISRec--MFilter-blue'></a>

Haidong Xin<sup>1</sup>,
Zhenghao Liu<sup>1†</sup>,
Sen Mei<sup>2</sup>,
Yukun Yan<sup>2</sup>,
Shi Yu<sup>2</sup>,
Shuo Wang<sup>2</sup>,
Zulong Chen<sup>3</sup>,
Yu Gu<sup>1</sup>,
Ge Yu<sup>1</sup>,
Chenyan Xiong<sup>4</sup>

<sup>1</sup>Northeastern University, <sup>2</sup>Tsinghua University, <sup>3</sup>Alibaba Group, <sup>4</sup>Carnegie Mellon University

</h5>
</div>

## Introduction

LISRec addresses the challenge of noisy data in sequential recommendation by constructing a user-interacted item graph. It leverages item similarities derived from their text representations to extract the maximum connected subgraph, effectively denoising the items a user has interacted with. LISRec demonstrates strong generalization capabilities by enhancing both item ID-based and text-based recommendation models.

![](figs/model.jpg)

## Requirements

### 1. Python Environment:

Install the following packages using Pip or Conda under this environment.

```shell
pip install -r requirements.txt
```

### 2. Install Openmatch.

Refer to [https://github.com/OpenMatch/OpenMatch](https://github.com/OpenMatch/OpenMatch) for detailed instructions.

```bash
git clone https://github.com/OpenMatch/OpenMatch.git
cd OpenMatch
pip install -e .
```

### 3. Pretrained T5 weights.

Download pretrained T5 weights from Hugging Face.

```bash
git lfs install
git clone https://huggingface.co/google-t5/t5-base
```

*Note:* Ensure that `git lfs` is properly installed. You may need to run `git lfs install` before cloning the T5 weights.

## Reproduction Guide

This section provides a step-by-step guide to reproduce the LISRec results.

> Warning: The model pre-training and fine-tuning process requires a lot of GPU resources, and the embedding vectors also take up a lot of space. Please make sure you have sufficient GPU resources and hard disk storage space.

### 1. Dataset Preprocessing

We utilize the Amazon Product 2014 and Yelp 2020 datasets. Download the original data from:

- [Amazon Product 2014](https://jmcauley.ucsd.edu/data/amazon/index_2014.html)
- [Yelp 2020](https://business.yelp.com/data/resources/open-dataset/)

The following example uses the Amazon Beauty dataset.

#### 1.1. Download and Prepare Amazon Beauty Dataset:

```bash
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz
```

#### 1.2. Unzip the Metadata File:

```bash
gzip -d meta_Beauty.json.gz
```

#### 1.3. Organize Files:

```bash
mkdir data
mv ratings_Beauty.csv data/
mv meta_Beauty.json data/
```

#### 1.4. Process Raw Data for Recbole:

```bash
mkdir dataset
bash scripts/process_origin.sh
```

#### 1.5. Extract and Process Required Data:

```bash
bash scripts/process_beauty.sh
```

### 2. Data Preprocessing for Training $\text{M}_{Filter}$

Before proceeding, process all four original datasets as described above to obtain the atomic files. Then, construct the mixed pretraining data for $\text{M}_{Filter}$ according to your desired proportions.

#### 2.1. Construct Training and Test Data using Recbole:

```bash
bash scripts/gen_dataset.sh
```

#### 2.2. Generate Item Representations using $\text{M}_{Rec}$:

```bash
bash scripts/gen_pretrain_items.sh
```

#### 2.3. Sample Training Data for $\text{M}_{Filter}$:

For $\text{M}_{Filter}$ training data construction, we sampled the four datasets with balance. For each dataset, we selected the number of items corresponding to the dataset with the largest number of training samples and then randomly supplemented the datasets with insufficient training data:

```bash
python src/sample_train.py
```

#### 2.4. Sample Validation Data:

Similarly, we selected the number of training samples from the dataset with the fewest training items in each case to serve as the validation set:

```bash
python src/sample_valid.py
```

#### 2.5. Construct Pretraining Data for Sampled Items:

```bash
bash scripts/build_pretrain.sh
```

#### 2.6. Merge Training and Validation Data:

```bash
python src/merge_json.py
```

### 3. Pretraining for $\text{M}_{Filter}$

Pretrain the T5 model using next item prediction (NIP) and mask item prediction (MIP) tasks.

```bash
bash scripts/train_mfilter.sh
```

Adjust training parameters based on your GPU device. Select the checkpoint with the lowest evaluation loss as the final $\text{M}_{Filter}$ .

### 4. Generate Embedding Representations using \text{M}_{Filter}

Save the item embedding representations to avoid redundant calculations.

```bash
mkdir embedding
bash scripts/gen_gembeddings.sh
```

### 5. Denoise Dataset by Calculating the Maximum Connected Subgraph

Embed the nodes into an undirected graph and use BFS to calculate the maximum connected subgraph.

```bash
bash scripts/build_graph.sh
```

Copy the original item information file to the denoised data folder.

```bash
cp dataset/beauty/beauty.item dataset/beauty_filtered/
mv dataset/beauty_filtered/beauty.item dataset/beauty_filtered/beauty_filtered.item
```

### 6. Build Standardized Training Data for $\text{M}_{Rec}$ using Recbole

```bash
bash scripts/gen_dataset.sh
bash scripts/gen_train_items.sh
bash scripts/build_train.sh
```

### 7. Training $\text{M}_{Rec}$

```bash
bash scripts/train_mrec.sh
```

### 8. Evaluate $\text{M}_{Rec}$

```bash
bash scripts/eval_mrec.sh
```

### 9. Test $\text{M}_{Rec}$

```bash
bash scripts/test_mrec.sh
```

## Acknowledgement

- [OpenMatch](https://github.com/OpenMatch/OpenMatch): We utilize OpenMatch to reproduce the $\text{M}_{Rec}$ module.
- [Recbole](https://github.com/RUCAIBox/RecBole): We leverage RecBole for dataset processing and baseline reproduction.

## Citation

If you find this work useful, please cite our paper and give us a shining star 🌟

```bibtex
@inproceedings{xin2026lisrec,
  title={LISRec: Modeling User Preferences with Learned Item Shortcuts for Sequential Recommendation},
  author={Xin, Haidong and Liu, Zhenghao and Mei, Sen and Yan, Yukun and Yu, Shi and Wang, Shuo and Xiong, Chenyan and Gu, Yu and Yu, Ge and Xiong, Chenyan},
  year={2026},
  url={}
}
```

## Contact

For questions, suggestions, or bug reports, please contact:

```
xinhaidong@stumail.neu.edu.cn
```
