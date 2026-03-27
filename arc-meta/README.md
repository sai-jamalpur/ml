# ARC-Meta

A meta-learning approach to solving the Abstraction and Reasoning Corpus (ARC) tasks using deep neural networks.

## Overview

**ARC-Meta** is a neural network-based solution for the ARC challenge—a benchmark of visual reasoning and abstract pattern recognition tasks. The project implements a meta-learning model architecture combining grid encoders, task encoders, and attention mechanisms to learn from few-shot examples and generalize to unseen tasks.

### What is ARC?

The Abstraction and Reasoning Corpus is a collection of visual reasoning tasks where models must:
- Learn from a small number of input-output grid examples
- Infer the underlying transformation rule
- Apply that rule to new test grids

Each task contains:
- **Training examples**: Input-output grid pairs demonstrating the transformation
- **Test examples**: Input grids requiring the transformation to be applied

## Features

- **Few-shot learning**: Master new tasks with minimal training examples
- **Grid-based reasoning**: Handles variable-sized grid inputs (normalized to 30×30)
- **Positional encoding**: 2D sine-cosine positional encoding for spatial awareness
- **Efficient architecture**: Optimized with batch normalization and convolutional layers
- **Comprehensive training pipeline**: Includes logging, checkpointing, and hyperparameter optimization
- **Evaluation framework**: Dedicated evaluation scripts with detailed metrics

## Installation

### Prerequisites

- Python ≥ 3.12
- CUDA 11.8+ (GPU acceleration recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd arc-meta
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Start training the model with default hyperparameters:
```bash
python src/train.py
```

With custom hyperparameters:
```bash
python src/train.py \
  --batch_size 8 \
  --lr 1e-4 \
  --epochs 200 \
  --entropy_w 0.001 \
  --resume models/arc_v3_latest.pt
```

**Key training arguments:**
- `--batch_size`: Batch size (default: 4, optimal: 4)
- `--lr`: Learning rate (default: 3e-4)
- `--entropy_w`: Entropy regularization weight (default: 0.001, optimal: 0.001)
- `--epochs`: Number of training epochs (default: 200)
- `--resume`: Path to checkpoint for resuming training

### Evaluation

Evaluate the model on held-out test tasks:
```bash
python src/eval.py --model models/arc_v3_best.pt
```

### Hyperparameter Optimization

Run grid search for optimal hyperparameters:
```bash
python grid_search.py
```

### Demo & Testing

Test the model with example tasks:
```bash
python -m jupyter notebook src/test.ipynb
```

## Project Structure

```
arc-meta/
├── src/                          # Source code
│   ├── model.py                 # Core neural network architecture
│   ├── arc_dataset.py           # Dataset loading and preprocessing
│   ├── arc_dataloader.py        # PyTorch DataLoader configuration
│   ├── train.py                 # Training loop
│   ├── eval.py                  # Evaluation script
│   ├── hpo.py                   # Hyperparameter optimization
│   ├── train_utils.py           # Utility functions for training
│   └── test.ipynb               # Interactive testing notebook
│
├── data/
│   ├── training/                # Training task files (JSON format)
│   └── evaluation/              # Evaluation task files (JSON format)
│
├── models/                       # Trained model checkpoints
│   ├── arc_v3_best.pt          # Best model (highest validation accuracy)
│   ├── arc_v3_latest.pt        # Latest checkpoint
│   └── arc_v3_epoch_*.pt       # Epoch checkpoints
│
├── logs/                         # Training logs and experiment records
│
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project metadata
└── README.md                    # This file
```

## Model Architecture

The ARC-Meta model consists of several key components:

### GridEncoder
- Embeds discrete cell values (0-9) into a dense latent space
- Applies convolutional blocks for local feature extraction
- Adds 2D positional encoding for spatial awareness

### TaskEncoder
- Processes support (training) examples to build a task representation
- Uses attention mechanisms to focus on relevant patterns
- Outputs a task-aware query context

### QueryProcessor
- Takes input grids and applies the learned transformation
- Incorporates task context from the TaskEncoder
- Predicts output grids with cell-level predictions

### Key Features
- **Batch Normalization**: Stabilizes training across diverse tasks
- **Positional Encoding**: Sine-cosine 2D positional embeddings
- **Entropy Regularization**: Encourages diversity in predictions

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `torchvision` | Computer vision utilities |
| `numpy` | Numerical computing |
| `matplotlib` | Visualization |
| `tqdm` | Progress bars |
| `einops` | Einstein operations for tensor manipulation |

## Training & Results

The model is trained on the ARC training dataset with:
- **Batch size**: 4
- **Learning rate**: 3e-4
- **Epochs**: 200
- **Optimization**: Adam with default parameters
- **Loss**: Cross-entropy with entropy regularization

**Best checkpoints**:
- `arc_v3_best.pt`: Best validation accuracy model
- `arc_v3_latest.pt`: Most recent training checkpoint

Training logs are saved to `logs/` with timestamps for tracking experiment history.

## Performance

Model performance is evaluated on the ARC evaluation set:
- Task-level accuracy
- Grid-level accuracy
- Detailed metrics saved to evaluation logs

## Development

### Running Tests
```bash
# Test individual components
python -c "from src.model import ARCModel; model = ARCModel(); print(model)"

# Full interactive testing
jupyter notebook src/test.ipynb
```

### Checkpointing
Models are automatically saved during training:
- Every epoch: `models/arc_v3_epoch_<N>.pt`
- Best model: `models/arc_v3_best.pt`
- Latest: `models/arc_v3_latest.pt`

## Configuration

All training parameters can be configured via command-line arguments. The optimal configuration discovered through hyperparameter search is:
- Batch size: 4
- Learning rate: 3e-4
- Entropy weight: 0.001

## Future Improvements

- [ ] Implement more sophisticated attention mechanisms
- [ ] Explore different positional encoding schemes
- [ ] Add explainability features for model decisions
- [ ] Support for larger grid sizes
- [ ] Multi-GPU training support
- [ ] Real-time inference API

## References

- [ARC Challenge](https://www.kaggle.com/c/abstraction-and-reasoning-corpus)
- [ARC Dataset Paper](https://arxiv.org/abs/1911.01547)

## License

[Add your license information here]

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or feedback, please open an issue in the repository.
