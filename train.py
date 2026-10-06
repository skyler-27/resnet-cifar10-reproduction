"""Train, select the best validation checkpoint, then evaluate test once."""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn

from data import CLASSES, loaders
from model import ResNet


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum = correct = total = 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        predictions = logits.argmax(1)
        n = labels.numel()
        loss_sum += loss.item() * n
        correct += (predictions == labels).sum().item()
        total += n
        confusion += torch.bincount((labels * 10 + predictions).cpu(),
                                    minlength=100).reshape(10, 10)
    return {'loss': loss_sum / total, 'accuracy': correct / total}, confusion


def plot_history(rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, axis in [('loss', axes[0]), ('accuracy', axes[1])]:
        for split in ('train', 'validation'):
            axis.plot([r['epoch'] for r in rows], [r[f'{split}_{key}'] for r in rows], label=split)
        axis.set(xlabel='Epoch', ylabel=key.capitalize())
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_confusion(matrix, output):
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(image, ax=ax)
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predicted', ylabel='True')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, choices=(18, 34), default=18)
    parser.add_argument('--no-residual', action='store_true')
    parser.add_argument('--no-augmentation', action='store_true')
    parser.add_argument('--optimizer', choices=('sgd', 'adam'), default='sgd')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('runs/resnet18'))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    train_loader, validation_loader, test_loader = loaders(
        args.data_dir, args.batch_size, args.seed, not args.no_augmentation,
        workers=args.workers)
    model = ResNet(args.depth, residual=not args.no_residual).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = (torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                 weight_decay=args.weight_decay) if args.optimizer == 'sgd'
                 else torch.optim.Adam(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay))
    milestones = sorted({max(1, int(args.epochs * fraction)) for fraction in (.5, .75)
                         if int(args.epochs * fraction) < args.epochs})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=.1)
    history = []
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics, _ = run_epoch(model, validation_loader, criterion, device)
        row = {'epoch': epoch, 'train_loss': train_metrics['loss'],
               'train_accuracy': train_metrics['accuracy'],
               'validation_loss': validation_metrics['loss'],
               'validation_accuracy': validation_metrics['accuracy']}
        history.append(row)
        if validation_metrics['accuracy'] > best_accuracy:
            best_accuracy = validation_metrics['accuracy']
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'validation_accuracy': best_accuracy},
                       args.out / 'best.pt')
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} train={train_metrics['accuracy']:.4f} "
              f"val={validation_metrics['accuracy']:.4f}", flush=True)
    with (args.out / 'history.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, args.out / 'curves.png')
    checkpoint = torch.load(args.out / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    test_metrics, confusion = run_epoch(model, test_loader, criterion, device)
    plot_confusion(confusion.numpy(), args.out / 'confusion.png')
    result = {'best_epoch': checkpoint['epoch'], 'validation_accuracy': best_accuracy,
              'test_loss': test_metrics['loss'], 'test_accuracy': test_metrics['accuracy']}
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
