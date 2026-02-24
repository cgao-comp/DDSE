
# DDSE-ACMMM2025

The code related to the paper below: Shenjie Jiang, Zhuoyu Wang, Xuecheng Wu, Hongru Ji, Mingxin Li, Xianghua Li, Chao Gao, DDSE: A decoupled dual-stream enhanced framework for multimodal sentiment analysis with text-centric SSM, The 33rd ACM International Conference on Multimedia (ACM MM25), 2025, 5893-5902



# Run
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

## Datasets

- **MOSI**
- **MOSEI**

You should put them in the `./dataset` folder, or modify the dataset path on your own in `config/config.json`.

## Download Links
Download processed MOSI, MOSEI datasets from the following links:
- [MOSI dataset](https://drive.google.com/drive/folders/1BBadVSptOe4h8TWchkhWZRLJw8YG_aEi?usp=sharing)
- [MOSEI dataset](https://drive.google.com/drive/folders/1BBadVSptOe4h8TWchkhWZRLJw8YG_aEi?usp=sharing)



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

## Execution
Before running, you need to set the necessary parameters in `./config/config.json`. Run logs and run results are saved in the `./log` and `./result/normal` directories by default, respectively.

- **Training**
You can first set the `dataset_name='mosi'` or `dataset_name='mosei'` in `train.py`, and then run:
```bash
python train.py
```
By default, the trained model will be saved in the `./pt` directory. You can change this in `train.py`.

- **Testing**
You can first set the `dataset_name='mosi'` or `dataset_name='mosei'` in `test.py`, and set the path of the trained model in `run.py`. Then test the trained model:
```bash
python test.py
```


# Project Structure

```
DDSE/
│
├── train.py                  # Training entry
├── test.py                   # Evaluation script
├── model/
│   ├── model_ddse.py         # Core DDSE architecture
│   └── modules/              # TC-Mamba and auxiliary modules
├── config/config.json        # Runtime configuration
├── dataset/                  # Processed MOSI/MOSEI datasets
├── utils/                    # Logging and evaluation utilities
├── requirements.txt
└── run.sh
```



# Acknowledgment

We acknowledge the contributions of open-source implementations such as **DMD** ([GitHub Repository](https://github.com/mdswyz/DMD/tree/main)) , **DepMamba** ([GitHub Repository](https://github.com/Jiaxin-Ye/DepMamba)) , and **DLF** ([GitHub Repository](https://github.com/pwang322/DLF)) , which inspired parts of our experimental setup.


# Reference

If you make advantage of DDSE in your research, please cite the following in your manuscript:

```bibtex
@inproceedings{jiang2025ddse,
  title     = {{DDSE}: A Decoupled Dual-Stream Enhanced Framework for Multimodal Sentiment Analysis with Text-Centric {SSM}},
  author    = {Jiang, Shenjie and Wang, Zhuoyu and Wu, Xuecheng and Ji, Hongru and Li, Mingxin and Li, Xianghua and Gao, Chao},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  pages     = {5893--5902},
  year      = {2025}
}
```


