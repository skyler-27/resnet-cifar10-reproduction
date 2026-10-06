# ResNet on CIFAR-10

An educational PyTorch implementation of the ResNet-18 and ResNet-34 basic-block architectures for 32×32 CIFAR-10 images. The 3×3 stem replaces the ImageNet model's 7×7 stem and max pooling. The comparison with `--no-residual` removes skip additions while keeping the same convolutional layers. This is a **CIFAR adaptation and ablation**, not a numerical reproduction of the original paper's ImageNet results.

## Study checkpoints

1. Read `BasicBlock.forward` in `model.py`: explain why adding `shortcut(x)` still works when a block downsamples from 32×32 to 16×16. The 1×1 projection in `self.shortcut` is the key.
2. Run the three tests. In `test_residual_connection_changes_output`, toggle `block.residual` and inspect the two outputs. The switch changes only the shortcut addition.
3. Read `split_indices` and `loaders` in `data.py`: verify that training and validation use different indices, validation has no random augmentation, and the official test partition is separate.
4. Run the one-epoch smoke check, then a full ResNet-18 baseline. Open `curves.png` and look for a gap between train and validation accuracy.
5. Run the plain network with the **same seed and settings**. Explain any difference using the curves and validation results; do not assume residual connections must win in every short run.

The project is complete as a portfolio experiment when the baseline and at least one controlled ablation have finished, their real validation/test results are recorded below, and the comparison is explained in your own words.

## Setup

Use Python 3.10–3.12 in a virtual environment and install `requirements.txt`. The first training run downloads CIFAR-10 into `data/`. On a Mac with supported Apple silicon, training uses MPS; on a CUDA system, it uses CUDA; otherwise it uses CPU. Start with `--workers 0` if multiprocessing fails.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pytest
python -m pytest -q
```

## Train and evaluate

```bash
python train.py --depth 18 --epochs 50 --out runs/resnet18
```

`train.py` uses 45,000 training images and 5,000 validation images from CIFAR-10's training partition. It selects the checkpoint using validation accuracy. It then evaluates the separate 10,000-image test partition once. Each run writes `best.pt`, `history.csv`, `curves.png`, `confusion.png`, and `result.json` to its output directory.

For a quick pipeline check, run `python train.py --epochs 1 --out runs/smoke`. A single epoch is not evidence of model performance.

## Ablation plan

Keep seed, split, epoch count, batch size, and learning rate fixed when comparing skip connections or augmentation. Use a fresh `--out` path for each run.

```bash
python train.py --depth 18 --epochs 50 --out runs/resnet18
python train.py --depth 18 --epochs 50 --no-residual --out runs/plain18
python train.py --depth 18 --epochs 50 --no-augmentation --out runs/no-augmentation
python train.py --depth 34 --epochs 50 --out runs/resnet34
```

For Adam, tune the learning rate on **validation** only, e.g. `--optimizer adam --lr 0.001`, then run test evaluation with the selected setting. Do not compare Adam at `0.1` to SGD at `0.1` and treat that as a fair optimizer comparison.

| Run | Residual | Augmentation | Best validation accuracy | Test accuracy |
| --- | --- | --- | --- | --- |
| ResNet-18 | Yes | Yes | 93.62% | 92.95% |
| Plain-18 | No | Yes | 91.22% | 91.39% |
| ResNet-18 | Yes | No | Pending experiment | Pending experiment |
| ResNet-34 | Yes | Yes | Pending experiment | Pending experiment |

These results were reported from a Windows WSL run on an NVIDIA GeForce RTX 5060 with seed 42, 50 epochs, and batch size 64. The commands were `python train.py --depth 18 --epochs 50 --batch-size 64 --out runs/resnet18` and the same command with `--no-residual --out runs/plain18`. The residual model was 2.40 percentage points higher on validation and 1.56 points higher on the test set in this single-seed comparison. The reported output files are saved as [ResNet-18](results/resnet18_result.json) and [Plain-18](results/plain18_result.json) results. Repeat with more seeds before drawing a general conclusion. Record the runtime alongside results, and include only verified metrics in a résumé or GitHub description.

## References

- [He et al., Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [PyTorch CIFAR-10 tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial)
- [Torchvision CIFAR-10 dataset](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)
