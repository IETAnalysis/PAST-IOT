# PAST-IOT
## PAST-IoT: Self-Supervised IoT Intrusion Detection via Pattern-Aware Anomaly Injection and Spectral-Temporal Dual-Tower Decomposition

We understand that validating the model architecture and training process is crucial for peer review.
Therefore, we provide:

Project Structure
feature_ext.py: PCAP parsing and multi-processing feature extraction; generates .pt datasets.

model.py: Core network architecture (RevIN, Dual-tower backbone, Multi-scale CNN, Multi-task output heads).

pretrain.py: Self-supervised pre-training script (Contrastive loss, Reconstruction loss, Masked phase prediction).

fintest.py: Downstream fine-tuning and evaluation script (Supports dynamic sampling, early stopping, and confusion matrix output).

README.md: Project documentation.



## Environment Setup
Python 3.8+ is recommended. Install the main dependencies using:
```bash
pip install torch torchvision numpy pandas scikit-learn scapy tqdm
```

## Data Preparation & Feature Extraction
The pre-training phase utilizes normal traffic data injected with physical anomalies to perform multi-task learning (Contrastive Learning + Masked Recovery + Reconstruction).
```python
RAW_P = "/path/to/pretrain/pcap" # Raw data for pre-training
OUT_P = "/path/to/pretrain/pt"   # Feature output for pre-training

RAW_B = "/path/to/finetune/pcap" # Raw data for fine-tuning
OUT_B = "/path/to/finetune/pt"   # Feature output for fine-tuning
```
##Fine-tuning & Evaluation
Fine-tune the model on datasets with real attack labels. The script includes a built-in WeightedRandomSampler to mitigate extreme class imbalance between positive and negative samples.
```python
PROCESSED_DATA_B = "/path/to/finetune/pt"
PRETRAINED_PATH = "pretrained_encoder.pth" 
SAVE_FINE_TUNED = "finetuned_model.pth"
Modify hyperparameters such as LABEL_RATIO and EPOCHS as needed.
```
## E-mail
If you have any question, please feel free to contact us by e-mail (jiangtaozhai@nuist.edu.cn).
