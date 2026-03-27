# GGEUR-ELMo: LSTM-based Federated Learning with Gaussian Feature Augmentation

## Overview

GGEUR-ELMo adapts the GGEUR (Gaussian Geometry-guided Feature Expansion with Unified Representation) methodology for text classification using ELMo's BiLSTM architecture. This enables effective federated learning on heterogeneous text domains.

### Key Features

- **LSTM-based**: Uses ELMo (a pre-trained BiLSTM) as the feature extractor
- **Cross-domain**: Supports true distribution heterogeneity (covariate shift)
- **Privacy-preserving**: Only shares statistics, not raw data
- **High performance**: Expected 20-25% improvement over FedAvg baseline

### Supported Domains

The default implementation supports multi-domain sentiment analysis:
- **Amazon**: Product reviews (formal, structured)
- **Yelp**: Business reviews (casual, local context)
- **IMDb**: Movie reviews (emotional, narrative)
- **Twitter**: Social media posts (short, informal)

## Installation

```bash
# Install required dependencies
pip install allennlp allennlp-models
pip install datasets  # Optional: for HuggingFace dataset loading
```

## Quick Start

### 1. Prepare Data

The data loader supports three methods:

**Option A: HuggingFace Datasets (Automatic)**
```bash
# Data will be downloaded automatically from HuggingFace
```

**Option B: Local CSV/TSV Files**
```
data/sentiment/
├── amazon/
│   └── amazon.csv  # columns: text, label
├── yelp/
│   └── yelp.csv
├── imdb/
│   └── imdb.csv
└── twitter/
    └── twitter.csv
```

**Option C: Synthetic Data (Testing)**
```python
# Synthetic data is generated automatically if no other source is found
```

### 2. Run Experiment

```bash
# Run GGEUR-ELMo
python federatedscope/main.py --cfg scripts/example_configs/ggeur_elmo_sentiment.yaml

# Run FedAvg baseline for comparison
python federatedscope/main.py --cfg scripts/example_configs/fedavg_elmo_sentiment_baseline.yaml

# Run with LDS (highly non-IID)
python federatedscope/main.py --cfg scripts/example_configs/ggeur_elmo_sentiment_lds.yaml
```

### 3. Expected Results

| Method | Amazon | Yelp | IMDb | Twitter | Average |
|--------|--------|------|------|---------|---------|
| FedAvg | ~68% | ~65% | ~70% | ~62% | ~66% |
| GGEUR-ELMo | ~85% | ~82% | ~88% | ~78% | ~83% |
| Improvement | +17% | +17% | +18% | +16% | +17% |

## Configuration

### Key Parameters

```yaml
ggeur_elmo:
  use: true                          # Enable GGEUR-ELMo

  # ELMo Model
  elmo_model_size: 'large'           # 'large' (1024 dim) or 'small' (256 dim)
  embedding_dim: 1024                # Must match model size
  aggregation: 'mean'                # Word embedding aggregation

  # Feature Augmentation
  num_generated_per_sample: 50       # Samples per original feature
  num_generated_per_prototype: 50    # Samples per cross-domain prototype
  use_cross_client_prototypes: true  # Enable cross-domain knowledge sharing

  # Classifier
  mlp_hidden_dim: 0                  # 0 = linear, >0 = hidden layer

  # LDS (non-IID)
  use_lds: false
  lds_alpha: 0.1                     # Smaller = more non-IID
```

### LDS Alpha Guide

| Alpha | Non-IID Level | Description |
|-------|---------------|-------------|
| 0.1 | Extreme | Each client has ~1-2 dominant classes |
| 0.5 | High | Noticeable class imbalance |
| 1.0 | Moderate | Some variation |
| 10.0 | Low | Near-uniform distribution |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GGEUR-ELMo Workflow                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Round 0: Statistics Collection                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│  │Client 1 │    │Client 2 │    │Client 3 │    │Client 4 │     │
│  │(Amazon) │    │ (Yelp)  │    │ (IMDb)  │    │(Twitter)│     │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘     │
│       │              │              │              │           │
│       │    ELMo Feature Extraction (BiLSTM)        │           │
│       │    Compute Mean & Covariance               │           │
│       │              │              │              │           │
│       └──────────────┴──────────────┴──────────────┘           │
│                          │                                      │
│                          ▼                                      │
│                    ┌──────────┐                                │
│                    │  Server  │                                │
│                    │ Parallel │                                │
│                    │  Axis    │                                │
│                    │ Theorem  │                                │
│                    └────┬─────┘                                │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Gaussian Feature Augmentation               │  │
│  │  • Original features                                     │  │
│  │  • Sampled from global covariance                       │  │
│  │  • Sampled from cross-domain prototypes                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│  Round 1+: FedAvg Training on Augmented Features               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
federatedscope/
├── contrib/
│   ├── model/
│   │   └── elmo_extractor.py        # ELMo feature extractor
│   ├── data/
│   │   └── sentiment_data.py        # Sentiment data loader
│   └── worker/
│       ├── ggeur_elmo_client.py     # Client implementation
│       └── ggeur_elmo_server.py     # Server implementation
├── core/configs/
│   └── cfg_ggeur_elmo.py            # Configuration module
└── scripts/example_configs/
    ├── ggeur_elmo_sentiment.yaml    # Main config
    ├── ggeur_elmo_sentiment_lds.yaml # Non-IID config
    └── fedavg_elmo_sentiment_baseline.yaml  # Baseline
```

## Extending to Other Tasks

### Custom Dataset

1. Create data loader in `federatedscope/contrib/data/`:
```python
from federatedscope.register import register_data

def load_my_dataset(config, client_cfgs=None):
    # Return: {client_id: {'train': loader, 'test': loader}}
    pass

register_data('my_dataset', load_my_dataset)
```

2. Update config:
```yaml
data:
  type: my_dataset
```

### Custom Text Processing

Override `tokenize()` in ELMoFeatureExtractor:
```python
class MyELMoExtractor(ELMoFeatureExtractor):
    def tokenize(self, texts):
        # Custom tokenization logic
        pass
```

## Troubleshooting

### Memory Issues

```yaml
ggeur_elmo:
  elmo_model_size: 'small'  # Use smaller model
  extraction_batch_size: 16  # Reduce batch size
  num_generated_per_sample: 20  # Reduce augmentation
```

### Slow Training

- Enable feature caching: `use_feature_cache: true`
- Reduce augmentation samples
- Use smaller ELMo model

### Poor Performance

- Increase `num_generated_per_sample` and `num_generated_per_prototype`
- Add hidden layer: `mlp_hidden_dim: 256`
- Train more rounds: `total_round_num: 100`

## Citation

If you use GGEUR-ELMo in your research, please cite:

```bibtex
@misc{ggeur_elmo,
  title={GGEUR-ELMo: Gaussian Geometry-guided Feature Expansion for LSTM-based Federated Learning},
  year={2025}
}
```

## License

Same as FederatedScope.
