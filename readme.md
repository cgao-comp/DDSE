# DDSE: A Decoupled Dual-Stream Enhanced Framework for Multimodal Sentiment Analysis with Text-Centric SSM
> This paper introduces DDSE, a decoupled dual-stream framework that leverages a text-centric state-space mechanism to achieve effective multimodal fusion for sentiment analysis.
## Abstract
Multimodal Sentiment Analysis (MSA) aims to identify sentiment polarity and intensity in media. Current methods typically employ a two-stage pipeline: extracting features from each modality, then predicting sentiment based on fused representations. However, most fusion strategies align features from different modalities in a single step, leading to conflicts during cross-modal interactions and hindering the modeling of hierarchical sentiment dependencies. Additionally, existing methods often overlook the dominant role of textual modality in high level latent fusion space, causing explicit linguistic sentiment cues to be obscured by redundant information. To address these issues, DDSE (Decoupled Dual-Stream Enhanced framework) is proposed in this work, which decouples features into public and private representations for improved feature enhancement and cross-modal interaction. The proposed TC-Mamba module enables progressive cross-modal interactions within shared state transition matrices under a text-guided fusion paradigm, effectively preserving sentiment cues and minimizing redundancy. Additionally, DDSE adopts a multi-task learning strategy to further enhance overall performance. Extensive experiments on the MOSI and MOSEI datasets demonstrate that DDSE achieves state-of-the-art results, with Acc-5 improvements of 3.06% and 0.1%, respectively, underscoring its effectiveness in MSA. Ablation studies confirm the critical contributions of each component within the framework.

## Project Structure

The project is organized as follows:

* **`train.py`** — Main training script for DDSE, handling data loading, optimization, and checkpoint saving.
* **`test.py`** — Evaluation script for testing pre-trained models on MOSI/MOSEI datasets.
* **`model/`**

  * `model_ddse.py`: Implementation of the core DDSE architecture.
  * `modules/`: Contains submodules for feature enhancement, attention mechanisms, and the TC-Mamba block.
* **`config/config.json`** — Configuration file for dataset paths, hyperparameters, and runtime settings.
* **`dataset/`** — Directory for processed datasets (MOSI, MOSEI).
* **`utils/`** — Helper functions for logging, evaluation, and metric computation.
* **`requirements.txt`** — Required dependencies and library versions.
* **`run.sh`** — Example shell script for setting up and running experiments.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/jiangshenjie/DDSE.git
   cd DDSE
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv ddse_env
   source ddse_env/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Prerequisites
Python version: 3.8.20

PyTorch version: 2.0.1

CUDA version: 11.7

### Default Settings
Before running, you need to set the necessary parameters in `./config/config.json`. Run logs and run results are saved in the `./log` and `./result/normal` directories by default, respectively.

### Run the Code
- Install the necessary packages
```
pip install -r requirements.txt
```

- Training

You can first set the dataset_name='mosi' or dataset_name='mosei' in `train.py`, and then run:
```
python train.py
```
By default, the trained model will be saved in the `./pt` directory. You can change this in `train.py`.
- Testing

You can first set the dataset_name='mosi' or dataset_name='mosei' in `test.py`, and set the path of the trained model in `run.py`. Then test the trained model:
```
python test.py
```
## Datasets

- **MOSI**
- **MOSEI**


You should put them in the `./dataset` folder, or modify the dataset path on your own in `config/config.json`. 

### Download Links

Download processed MOSI, MOSEI datasets from the following links:

- [MOSI dataset](https://drive.google.com/drive/folders/1BBadVSptOe4h8TWchkhWZRLJw8YG_aEi?usp=sharing)
- [MOSEI dataset](https://drive.google.com/drive/folders/1BBadVSptOe4h8TWchkhWZRLJw8YG_aEi?usp=sharing)


## Evaluation
DDSE is evaluated on MOSI and MOSEI using standard metrics:

* **Accuracy (Acc-2 / Acc-5 / Acc-7)**
* **F1-score (F1)**
* **Mean Absolute Error (MAE)**
* **Correlation (Corr)**

**Ablation studies** confirm that removing TC-Mamba or decoupled streams significantly reduces performance, highlighting their contributions to feature disentanglement and sentiment retention.

## Results
DDSE consistently outperforms baseline models including MISA, Self-MM, and MAG-BERT, demonstrating superior sentiment representation and interpretability.

## Acknowledgments
This project builds upon prior work in **multimodal sentiment analysis**, **state-space modeling**, and **decoupled representation learning**.
We acknowledge the contributions of open-source implementations such as **DMD** ([GitHub Repository](https://github.com/mdswyz/DMD/tree/main)) , **DepMamba** ([GitHub Repository](https://github.com/Jiaxin-Ye/DepMamba)) , and **DLF** ([GitHub Repository](https://github.com/pwang322/DLF)) , which inspired parts of our experimental setup.

## Citation
If you find this project useful for your research, please cite our paper:

```
@inproceedings{jiang2025ddse,
  title     = {{DDSE}: A Decoupled Dual-Stream Enhanced Framework for Multimodal Sentiment Analysis with Text-Centric {SSM}},
  author    = {Jiang, Shenjie and Wang, Zhuoyu and Wu, Xuecheng and Ji, Hongru and Li, Mingxin and Li, Xianghua and Gao, Chao},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  year      = {2025},
  pages     = {to appear}
}
```



