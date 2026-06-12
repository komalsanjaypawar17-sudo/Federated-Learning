# Federated-Learning

# Federated Predictive Maintenance — Japanese Automotive Plants

> Simulating privacy-preserving machine failure prediction across N distributed factory floors using the AI4I 2020 Predictive Maintenance Dataset.

---

## Overview

This project demonstrates **federated learning** for industrial predictive maintenance, inspired by the **Toyota Production System (TPS)**. Instead of centralizing sensitive sensor data from multiple plants, each factory trains a local model on its own shard, and only model updates are shared — preserving data sovereignty across competing manufacturers.

**Dataset:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — UCI ML Repository, ID 601  
**Task:** Binary classification — predict `Machine failure` (0 = healthy, 1 = failure)

---

## Simulated Factory Clients

Each "client" represents a real Japanese automotive assembly plant:

| Client ID | Plant | Manufacturer |
|-----------|-------|--------------|
| 0 | Motomachi Plant | Toyota |
| 1 | Tsutsumi Plant | Toyota |
| 2 | Suzuka Factory | Honda |
| 3 | Oppama Plant | Nissan |
| 4 | Gunma Main Plant | Subaru |

Additional clients (if `N > 5`) are assigned sequentially.

---

## Features

| Column | Description |
|--------|-------------|
| `Air temperature [K]` | Ambient temperature around the machine |
| `Process temperature [K]` | Temperature of the active manufacturing process |
| `Rotational speed [rpm]` | Spindle / motor rotational speed |
| `Torque [Nm]` | Torque applied during operation |
| `Tool wear [min]` | Accumulated tool wear time |
| **`Machine failure`** | **Target label** (0 = no failure, 1 = failure) |

---

## Project Structure

```
.
├── data/
│   └── ai4i2020.csv              # Downloaded dataset (auto-generated)
├── clients/
│   └── client_{0..N-1}.csv       # Per-plant shards
├── split_data.py                 # Downloads and shards the dataset
├── train_local.py                # Local training per client
├── federated_server.py           # FedAvg aggregation server
├── evaluate.py                   # Global model evaluation
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download and shard the dataset

```bash
python split_data.py --n_clients 5 --strategy iid
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--n_clients` | `5` | Number of factory clients (plants) |
| `--strategy` | `iid` | Sharding strategy: `iid` or `non_iid` |
| `--seed` | `42` | Random seed for reproducibility |
| `--output_dir` | `clients/` | Output directory for client shards |

> **IID** (Independent & Identically Distributed): rows are randomly shuffled before splitting — each plant sees a representative cross-section of failure types.  
> **Non-IID**: shards are split by failure type or operational regime — more realistic, as different plants run different tooling.

### 3. Train local models

```bash
# Train all clients
for i in $(seq 0 4); do
    python train_local.py --client_id $i
done
```

### 4. Run federated aggregation

```bash
python federated_server.py --rounds 10 --n_clients 5
```

### 5. Evaluate global model

```bash
python evaluate.py
```

---

## Data Splitting Logic

```
Total rows (10,000)
        │
        ▼
  Shuffle (seed=42)
        │
   ┌────┴────────────────────────┐
   │   Split into N equal shards │
   └─────────────────────────────┘
        │
  ┌─────┼─────┐─────┐─────┐
  ▼     ▼     ▼     ▼     ▼
 P0    P1    P2    P3    P4
Toyota Toyota Honda Nissan Subaru
```

Each shard retains the original failure rate (~3.4% positive class) in IID mode. Non-IID mode intentionally creates label skew across plants.

---

## Class Imbalance

The dataset is heavily imbalanced (~96.6% healthy, ~3.4% failure). Recommended mitigations:

- `class_weight='balanced'` in sklearn estimators
- SMOTE oversampling on local client data
- Weighted cross-entropy loss in PyTorch/TF models
- Evaluate with **F1-score / AUC-ROC**, not raw accuracy

---

## Federated Learning Protocol

This project implements **FedAvg** (McMahan et al., 2017):

1. Server initializes global model weights `w₀`
2. Each round `t`:
   - Server broadcasts `wₜ` to all clients
   - Each client trains locally for `E` epochs → produces `wₜᵢ`
   - Server aggregates: `wₜ₊₁ = Σ (nᵢ / n) · wₜᵢ`
3. Repeat for `R` rounds

No raw sensor data ever leaves the plant — only model weight deltas are transmitted.

---

## Toyota Production System Analogy

| TPS Concept | FL Equivalent |
|-------------|---------------|
| Kaizen (continuous improvement) | Iterative federated rounds |
| Jidoka (autonomation) | Local anomaly detection per client |
| Andon (stop-the-line signal) | Failure prediction triggering maintenance |
| Just-in-Time | Predictive maintenance replacing scheduled downtime |
| Genchi Genbutsu (go & see) | On-device learning from local sensor streams |

---

## Requirements

```
ucimlrepo>=0.0.3
pandas>=1.5
scikit-learn>=1.2
numpy>=1.23
imbalanced-learn>=0.10   # for SMOTE
torch>=2.0               # optional, for neural net baseline
flwr>=1.5                # optional, for production FL framework
```

---

## References

- Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
- McMahan, B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS.
- Liker, J. K. (2004). *The Toyota Way*. McGraw-Hill.

---

## License

Dataset: CC BY 4.0 (UCI ML Repository)  
Code: MIT
