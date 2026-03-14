"""
============================================================================
DeepRx: Training Pipeline — Research Publication Version
============================================================================
Improvements over prototype:
    1. Longer training (10,000 steps)
    2. Larger dataset (10,000 samples)
    3. Larger batch size (16)
    4. Proper LR schedule with cosine annealing
    5. Gradient accumulation support
    6. Reproducible with seed control
    7. Comprehensive logging
============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os
import json
import math
import random
import numpy as np
from typing import Dict, Optional, Tuple

from deeprx_model import (
    DeepRx, DeepRxLoss, compute_ber, MODULATION_CONFIG
)
from data_generator import DeepRxDataset
from traditional_receiver import TraditionalReceiver


# ═══════════════════════════════════════════════════════════════════════════
#  Reproducibility
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"  Random seed set to: {seed}")


# ═══════════════════════════════════════════════════════════════════════════
#  Learning Rate Scheduler
# ═══════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing schedule.
    Better convergence than linear decay for longer training.
    """

    def __init__(self, optimizer, max_lr=1e-3, min_lr=1e-6,
                 total_steps=10000, warmup_steps=500):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def get_lr(self):
        step = self.current_step
        if step < self.warmup_steps:
            return self.max_lr * (step / max(self.warmup_steps, 1))
        else:
            progress = (step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress))

    def step(self):
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.current_step += 1
        return lr


# ═══════════════════════════════════════════════════════════════════════════
#  Trainer — Publication Version
# ═══════════════════════════════════════════════════════════════════════════

class DeepRxTrainer:
    """Training pipeline optimized for research publication."""

    DEFAULT_CONFIG = {
        # Optimizer
        'max_lr': 1e-3,
        'min_lr': 1e-6,
        'weight_decay': 1e-4,
        'warmup_steps': 500,

        # Training
        'batch_size': 16,
        'total_steps': 10000,
        'val_every': 500,
        'log_every': 50,
        'grad_clip': 1.0,

        # Data
        'num_workers': 0,
        'modulation': '16QAM',

        # Saving
        'save_dir': 'checkpoints',
        'experiment_name': 'deeprx_publication',

        # Reproducibility
        'seed': 42,
    }

    def __init__(self, model, train_dataset, val_dataset,
                 config=None, device='cpu'):
        self.device = device
        self.model = model.to(device)

        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)

        # Set seed
        set_seed(self.config['seed'])

        self.criterion = DeepRxLoss()

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            drop_last=True,
            pin_memory=(device != 'cpu')
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            drop_last=False
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['max_lr'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
        )

        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            max_lr=self.config['max_lr'],
            min_lr=self.config['min_lr'],
            total_steps=self.config['total_steps'],
            warmup_steps=self.config['warmup_steps']
        )

        self.trad_receiver = TraditionalReceiver(
            self.config['modulation'], device
        )

        self.history = {
            'train_loss': [], 'train_ber': [],
            'val_loss': [], 'val_ber': [], 'val_ber_lmmse': [],
            'learning_rates': [], 'steps': [],
        }

        self.best_val_ber = float('inf')

        save_dir = os.path.join(
            self.config['save_dir'], self.config['experiment_name']
        )
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def train(self):
        cfg = self.config
        print("\n" + "=" * 70)
        print(f"{'DeepRx Training — Publication Version':^70}")
        print("=" * 70)
        print(f"  Device:       {self.device}")
        print(f"  Parameters:   {self.model.count_parameters():,}")
        print(f"  Batch size:   {cfg['batch_size']}")
        print(f"  Total steps:  {cfg['total_steps']}")
        print(f"  Max LR:       {cfg['max_lr']}")
        print(f"  Seed:         {cfg['seed']}")
        print(f"  Save dir:     {self.save_dir}")
        print("=" * 70)

        step = 0
        epoch = 0
        total_steps = cfg['total_steps']
        running_loss = 0.0
        running_ber = 0.0
        running_count = 0
        train_start = time.time()

        while step < total_steps:
            epoch += 1
            for batch_data in self.train_loader:
                if step >= total_steps:
                    break

                loss, ber = self._train_step(batch_data)
                lr = self.scheduler.step()

                running_loss += loss
                running_ber += ber
                running_count += 1
                step += 1

                if step % cfg['log_every'] == 0:
                    avg_loss = running_loss / running_count
                    avg_ber = running_ber / running_count
                    elapsed = time.time() - train_start
                    speed = step / max(elapsed, 1)
                    eta = (total_steps - step) / max(speed, 0.01)

                    self.history['train_loss'].append(avg_loss)
                    self.history['train_ber'].append(avg_ber)
                    self.history['learning_rates'].append(lr)
                    self.history['steps'].append(step)

                    print(
                        f"  Step {step:>6}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"BER: {avg_ber:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"ETA: {eta/60:.1f}min"
                    )
                    running_loss = 0.0
                    running_ber = 0.0
                    running_count = 0

                if step % cfg['val_every'] == 0:
                    val = self._validate()
                    self.history['val_loss'].append(val['loss'])
                    self.history['val_ber'].append(val['ber_deeprx'])
                    self.history['val_ber_lmmse'].append(val['ber_lmmse'])

                    print(f"\n  {'─'*60}")
                    print(f"  Validation at Step {step}")
                    print(f"    DeepRx Loss:  {val['loss']:.4f}")
                    print(f"    DeepRx BER:   {val['ber_deeprx']:.6f}")
                    print(f"    LMMSE  BER:   {val['ber_lmmse']:.6f}")

                    if val['ber_deeprx'] > 0:
                        gain = val['ber_lmmse'] / val['ber_deeprx']
                        print(f"    Gain:         {gain:.2f}x")

                    if val['ber_deeprx'] < self.best_val_ber:
                        self.best_val_ber = val['ber_deeprx']
                        self._save_checkpoint(step, is_best=True)
                        print(f"    ★ New best model!")

                    print(f"  {'─'*60}\n")

        self._save_checkpoint(step, is_best=False)
        total_time = time.time() - train_start

        print(f"\n{'='*70}")
        print(f"  Training Complete!")
        print(f"  Total time:    {total_time/60:.1f} minutes")
        print(f"  Best Val BER:  {self.best_val_ber:.6f}")
        print(f"{'='*70}\n")

        return self.history

    def _train_step(self, batch_data):
        self.model.train()
        inputs = batch_data['input'].to(self.device)
        targets = batch_data['target_bits'].to(self.device)
        data_mask = batch_data['data_mask'].to(self.device)
        bit_mask = batch_data['bit_mask'].to(
            self.device).unsqueeze(-1).unsqueeze(-1)

        logits = self.model(inputs)
        loss = self.criterion(logits, targets, data_mask, bit_mask)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config['grad_clip']
        )
        self.optimizer.step()

        with torch.no_grad():
            ber = compute_ber(logits, targets, data_mask, bit_mask)
        return loss.item(), ber

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_loss, total_deep, total_lmmse, n = 0, 0, 0, 0
        max_batches = 30

        for i, batch_data in enumerate(self.val_loader):
            if i >= max_batches:
                break

            inputs = batch_data['input'].to(self.device)
            targets = batch_data['target_bits'].to(self.device)
            data_mask = batch_data['data_mask'].to(self.device)
            bit_mask = batch_data['bit_mask'].to(
                self.device).unsqueeze(-1).unsqueeze(-1)

            logits = self.model(inputs)
            loss = self.criterion(logits, targets, data_mask, bit_mask)
            ber_deep = compute_ber(logits, targets, data_mask, bit_mask)
            ber_lmmse = self._compute_lmmse_ber(batch_data)

            total_loss += loss.item()
            total_deep += ber_deep
            total_lmmse += ber_lmmse
            n += 1

        return {
            'loss': total_loss / max(n, 1),
            'ber_deeprx': total_deep / max(n, 1),
            'ber_lmmse': total_lmmse / max(n, 1),
        }

    def _compute_lmmse_ber(self, batch_data):
        inputs = batch_data['input'].to(self.device)
        targets = batch_data['target_bits'].to(self.device)
        data_mask = batch_data['data_mask'].to(self.device)
        bit_mask = batch_data['bit_mask'].to(
            self.device).unsqueeze(-1).unsqueeze(-1)

        Nr = self.model.n_rx
        Nc = 2 * Nr + 1

        y_real = inputs[:, :Nr]
        y_imag = inputs[:, Nc:Nc+Nr]
        rx_grid = torch.complex(y_real, y_imag)

        xp_real = inputs[:, Nr:Nr+1]
        xp_imag = inputs[:, Nc+Nr:Nc+Nr+1]
        tx_pilots = torch.complex(xp_real, xp_imag)

        pilot_mask = (tx_pilots.abs() > 1e-6).float()[:1, :1]
        llrs = self.trad_receiver.process(rx_grid, tx_pilots, pilot_mask)
        return compute_ber(llrs, targets, data_mask, bit_mask)

    def _save_checkpoint(self, step, is_best=False):
        name = 'best_model.pt' if is_best else f'checkpoint_step{step}.pt'
        path = os.path.join(self.save_dir, name)
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_ber': self.best_val_ber,
            'config': self.config,
            'history': self.history,
        }, path)

        hist_path = os.path.join(self.save_dir, 'history.json')
        serializable = {k: [float(x) for x in v]
                        for k, v in self.history.items()}
        with open(hist_path, 'w') as f:
            json.dump(serializable, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    Nr = 2
    B_max = 8
    modulation = '16QAM'

    config = {
        'max_lr': 1e-3,
        'min_lr': 1e-6,
        'weight_decay': 1e-4,
        'warmup_steps': 300,
        'batch_size': 16,
        'total_steps': 10000,
        'val_every': 500,
        'log_every': 50,
        'grad_clip': 1.0,
        'num_workers': 0,
        'modulation': modulation,
        'save_dir': 'checkpoints',
        'experiment_name': 'deeprx_publication',
        'seed': 42,
    }

    print("\nCreating training dataset (10,000 samples)...")
    train_dataset = DeepRxDataset(
        n_samples=10000,
        n_rx_antennas=Nr,
        modulation=modulation,
        snr_range=(-4.0, 30.0),
        doppler_range=(0.0, 500.0),
        channel_profiles=['TDL_B', 'TDL_C', 'TDL_D', 'SIMPLE'],
        pilot_configs=['1_pilot_A', '1_pilot_B', '2_pilots_A', '2_pilots_B'],
        add_interference=False,
        device=device
    )

    print("Creating validation dataset (2,000 samples)...")
    val_dataset = DeepRxDataset(
        n_samples=2000,
        n_rx_antennas=Nr,
        modulation=modulation,
        snr_range=(0.0, 25.0),
        doppler_range=(10.0, 300.0),
        channel_profiles=['TDL_A', 'SIMPLE'],
        pilot_configs=['2_pilots_A'],
        add_interference=False,
        device=device
    )

    print("\nCreating DeepRx model...")
    model = DeepRx(n_rx_antennas=Nr, max_bits_per_symbol=B_max,
                   depth_multiplier=2)
    model.print_summary()

    trainer = DeepRxTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device
    )

    history = trainer.train()

    print("\n" + "=" * 70)
    print(f"{'Final Results':^70}")
    print("=" * 70)
    if history['val_ber']:
        print(f"  Final DeepRx BER:  {history['val_ber'][-1]:.6f}")
        print(f"  Final LMMSE BER:   {history['val_ber_lmmse'][-1]:.6f}")
        print(f"  Best DeepRx BER:   {trainer.best_val_ber:.6f}")
        if history['val_ber'][-1] > 0:
            gain = history['val_ber_lmmse'][-1] / history['val_ber'][-1]
            print(f"  Performance Gain:  {gain:.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
