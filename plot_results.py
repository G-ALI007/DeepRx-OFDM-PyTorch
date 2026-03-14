"""
============================================================================
DeepRx: Publication-Quality Plots with Confidence Intervals
============================================================================
"""

from traditional_receiver import TraditionalReceiver
from ofdm_system import (
    QAMModulator, OFDMTransmitter, OFDMReceiver,
    ChannelModel, add_awgn
)
from deeprx_model import (
    DeepRx, build_deeprx_input, compute_ber,
    create_pilot_mask, generate_qpsk_pilots, create_bit_mask
)
import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import math

import matplotlib
matplotlib.use('Agg')


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

STYLE = {
    'deeprx': {'color': '#1565C0', 'marker': 'o', 'linewidth': 2.5,
               'markersize': 8, 'label': 'DeepRx (Proposed)'},
    'lmmse':  {'color': '#C62828', 'marker': 's', 'linewidth': 2.5,
               'markersize': 8, 'label': 'LMMSE (Baseline)', 'linestyle': '--'},
}

FIG_DIR = 'figures'


def setup_style():
    plt.rcParams.update({
        'font.size': 13, 'axes.labelsize': 15, 'axes.titlesize': 16,
        'legend.fontsize': 12, 'figure.figsize': (8, 6), 'figure.dpi': 150,
        'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 2,
        'font.family': 'serif', 'savefig.format': 'pdf',
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    })


# ═══════════════════════════════════════════════════════════════════════════
#  Test Data Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_test_batch(batch_size, snr_db, doppler_hz,
                        channel_profile='TDL_B', pilot_config='2_pilots_A',
                        Nr=2, S=14, F=312, n_fft=512, cp_length=36,
                        device='cpu'):
    modulation = '16QAM'
    from deeprx_model import MODULATION_CONFIG
    bps = MODULATION_CONFIG[modulation]

    pilot_mask = create_pilot_mask(S, F, pilot_config, device)
    data_mask = 1.0 - pilot_mask
    bit_mask = create_bit_mask(modulation, 8, device)
    n_data = int(data_mask.sum().item())

    tx = OFDMTransmitter(F, n_fft, cp_length, S)
    rx_fe = OFDMReceiver(F, n_fft, cp_length, S)

    data_bits, data_syms = QAMModulator.bits_to_symbols(
        batch_size * n_data, modulation, device)
    data_syms = data_syms.reshape(batch_size, n_data)
    data_bits = data_bits.reshape(batch_size, n_data, -1)

    pilot_symbols = generate_qpsk_pilots(batch_size, S, F, pilot_mask, device)
    grid, target_bits, _ = tx.build_resource_grid(
        data_syms, pilot_symbols, pilot_mask, data_bits, bps)
    tx_signal = tx.modulate_ofdm(grid)
    sig_len = tx_signal.shape[1]
    sig_power = (tx_signal.abs() ** 2).mean().item()

    rx_waveforms = []
    for ant in range(Nr):
        ch = ChannelModel(channel_profile, doppler_hz, device=device)
        h_time, _ = ch.generate(batch_size, sig_len, S, n_fft, cp_length)
        rx_ant = ch.apply_channel(tx_signal, h_time)
        rx_ant = add_awgn(rx_ant, snr_db, sig_power)
        rx_waveforms.append(rx_ant)

    rx_multi = torch.stack(rx_waveforms, dim=1)
    rx_grid = rx_fe.demodulate(rx_multi, Nr)
    deeprx_input = build_deeprx_input(rx_grid, pilot_symbols)

    return {
        'input': deeprx_input, 'target_bits': target_bits,
        'data_mask': data_mask, 'bit_mask': bit_mask,
        'rx_grid': rx_grid, 'tx_pilots': pilot_symbols,
        'pilot_mask': pilot_mask,
    }


def evaluate_with_confidence(model, param_list, param_name,
                             fixed_params, n_trials=20,
                             batch_size=8, device='cpu'):
    """
    Evaluate with multiple trials and compute confidence intervals.

    Returns mean BER and 95% CI for both DeepRx and LMMSE.
    """
    trad_rx = TraditionalReceiver('16QAM', device)
    model.eval()

    results = {
        'params': param_list,
        'deeprx_mean': [], 'deeprx_ci': [],
        'lmmse_mean': [], 'lmmse_ci': [],
    }

    for param_val in param_list:
        ber_deep_trials = []
        ber_lmmse_trials = []

        for trial in range(n_trials):
            if param_name == 'snr':
                data = generate_test_batch(
                    batch_size, param_val, fixed_params.get('doppler', 50.0),
                    fixed_params.get('channel', 'TDL_B'),
                    fixed_params.get('pilot', '2_pilots_A'),
                    device=device)
            elif param_name == 'doppler':
                data = generate_test_batch(
                    batch_size, fixed_params.get('snr', 15.0), param_val,
                    fixed_params.get('channel', 'TDL_B'),
                    fixed_params.get('pilot', '2_pilots_A'),
                    device=device)

            with torch.no_grad():
                logits = model(data['input'])
            bd = compute_ber(logits, data['target_bits'],
                             data['data_mask'], data['bit_mask'])

            llrs = trad_rx.process(data['rx_grid'], data['tx_pilots'],
                                   data['pilot_mask'])
            bl = compute_ber(llrs, data['target_bits'],
                             data['data_mask'], data['bit_mask'])

            ber_deep_trials.append(bd)
            ber_lmmse_trials.append(bl)

        # Mean and 95% CI
        deep_arr = np.array(ber_deep_trials)
        lmmse_arr = np.array(ber_lmmse_trials)

        ci_factor = 1.96  # 95% confidence interval

        results['deeprx_mean'].append(deep_arr.mean())
        results['deeprx_ci'].append(
            ci_factor * deep_arr.std() / np.sqrt(n_trials))
        results['lmmse_mean'].append(lmmse_arr.mean())
        results['lmmse_ci'].append(
            ci_factor * lmmse_arr.std() / np.sqrt(n_trials))

        gain = lmmse_arr.mean() / max(deep_arr.mean(), 1e-8)
        print(f"    {param_name}={param_val:>6.1f} | "
              f"DeepRx: {deep_arr.mean():.4f}±{results['deeprx_ci'][-1]:.4f} | "
              f"LMMSE: {lmmse_arr.mean():.4f}±{results['lmmse_ci'][-1]:.4f} | "
              f"Gain: {gain:.2f}x")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 1: BER vs SNR with Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════════

def plot_ber_vs_snr(model, device='cpu'):
    print("\n  [1/6] BER vs SNR with 95% CI...")

    snr_list = list(range(0, 27, 3))
    results = evaluate_with_confidence(
        model, snr_list, 'snr',
        {'doppler': 50.0, 'channel': 'TDL_B', 'pilot': '2_pilots_A'},
        n_trials=25, batch_size=8, device=device
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    dm = np.array(results['deeprx_mean'])
    dc = np.array(results['deeprx_ci'])
    lm = np.array(results['lmmse_mean'])
    lc = np.array(results['lmmse_ci'])

    ax.semilogy(snr_list, dm, **STYLE['deeprx'])
    ax.fill_between(snr_list, np.maximum(dm-dc, 1e-5), dm+dc,
                    alpha=0.15, color=STYLE['deeprx']['color'])

    ax.semilogy(snr_list, lm, **STYLE['lmmse'])
    ax.fill_between(snr_list, np.maximum(lm-lc, 1e-5), lm+lc,
                    alpha=0.15, color=STYLE['lmmse']['color'])

    # Gain annotations
    for i, snr in enumerate(snr_list):
        if dm[i] > 0 and lm[i] / dm[i] > 1.05:
            ax.annotate(f'{lm[i]/dm[i]:.1f}×',
                        xy=(snr, dm[i]), xytext=(5, 12),
                        textcoords='offset points', fontsize=9,
                        color='green', fontweight='bold')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Uncoded BER')
    ax.set_title('BER vs SNR — DeepRx vs LMMSE Baseline\n'
                 '(16-QAM, Doppler=50 Hz, TDL-B, 2 Pilots, shaded=95% CI)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim([snr_list[0], snr_list[-1]])
    ax.set_ylim([5e-4, 1])
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_ber_vs_snr.pdf'))
    plt.close()
    print(f"    Saved: fig1_ber_vs_snr.pdf")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 2: BER vs Doppler with Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════════

def plot_ber_vs_doppler(model, device='cpu'):
    print("\n  [2/6] BER vs Doppler with 95% CI...")

    doppler_list = [10, 50, 100, 150, 200, 300, 400, 500]
    results = evaluate_with_confidence(
        model, doppler_list, 'doppler',
        {'snr': 15.0, 'channel': 'TDL_B', 'pilot': '2_pilots_A'},
        n_trials=25, batch_size=8, device=device
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    dm = np.array(results['deeprx_mean'])
    dc = np.array(results['deeprx_ci'])
    lm = np.array(results['lmmse_mean'])
    lc = np.array(results['lmmse_ci'])

    ax.semilogy(doppler_list, dm, **STYLE['deeprx'])
    ax.fill_between(doppler_list, np.maximum(dm-dc, 1e-5), dm+dc,
                    alpha=0.15, color=STYLE['deeprx']['color'])

    ax.semilogy(doppler_list, lm, **STYLE['lmmse'])
    ax.fill_between(doppler_list, np.maximum(lm-lc, 1e-5), lm+lc,
                    alpha=0.15, color=STYLE['lmmse']['color'])

    ax.set_xlabel('Maximum Doppler Shift (Hz)')
    ax.set_ylabel('Uncoded BER')
    ax.set_title('BER vs Doppler Shift — DeepRx vs LMMSE\n'
                 '(16-QAM, SNR=15 dB, TDL-B, 2 Pilots, shaded=95% CI)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_ber_vs_doppler.pdf'))
    plt.close()
    print(f"    Saved: fig2_ber_vs_doppler.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 3: Per-Bit BER
# ═══════════════════════════════════════════════════════════════════════════

def plot_per_bit_ber(model, device='cpu'):
    print("\n  [3/6] Per-Bit BER Analysis...")

    trad_rx = TraditionalReceiver('16QAM', device)
    bit_names = ['I-MSB\n($b_0$)', 'I-LSB\n($b_1$)',
                 'Q-MSB\n($b_2$)', 'Q-LSB\n($b_3$)']

    ber_deep, ber_lmmse = [], []
    ci_deep, ci_lmmse = [], []

    for bit_idx in range(4):
        single_mask = torch.zeros(1, 8, 1, 1, device=device)
        single_mask[0, bit_idx, 0, 0] = 1.0

        bd_trials, bl_trials = [], []
        for _ in range(30):
            data = generate_test_batch(8, 15.0, 50.0, device=device)
            with torch.no_grad():
                logits = model(data['input'])
            bd_trials.append(compute_ber(logits, data['target_bits'],
                                         data['data_mask'], single_mask))
            llrs = trad_rx.process(data['rx_grid'], data['tx_pilots'],
                                   data['pilot_mask'])
            bl_trials.append(compute_ber(llrs, data['target_bits'],
                                         data['data_mask'], single_mask))

        bd_arr, bl_arr = np.array(bd_trials), np.array(bl_trials)
        ber_deep.append(bd_arr.mean())
        ber_lmmse.append(bl_arr.mean())
        ci_deep.append(1.96 * bd_arr.std() / np.sqrt(30))
        ci_lmmse.append(1.96 * bl_arr.std() / np.sqrt(30))

        print(f"    Bit {bit_idx}: DeepRx={bd_arr.mean():.4f}±{ci_deep[-1]:.4f} | "
              f"LMMSE={bl_arr.mean():.4f}±{ci_lmmse[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(4)
    width = 0.35

    ax.bar(x - width/2, ber_deep, width, yerr=ci_deep,
           color=STYLE['deeprx']['color'], label='DeepRx', alpha=0.85,
           capsize=4, edgecolor='white')
    ax.bar(x + width/2, ber_lmmse, width, yerr=ci_lmmse,
           color=STYLE['lmmse']['color'], label='LMMSE', alpha=0.85,
           capsize=4, edgecolor='white')

    for i in range(4):
        ax.text(x[i]-width/2, ber_deep[i]+ci_deep[i]+0.0005,
                f'{ber_deep[i]:.4f}', ha='center', fontsize=9)
        ax.text(x[i]+width/2, ber_lmmse[i]+ci_lmmse[i]+0.0005,
                f'{ber_lmmse[i]:.4f}', ha='center', fontsize=9)

    ax.set_ylabel('BER')
    ax.set_title(
        'Per-Bit BER — 16-QAM\n(SNR=15 dB, Doppler=50 Hz, error bars=95% CI)')
    ax.set_xticks(x)
    ax.set_xticklabels(bit_names)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_per_bit_ber.pdf'))
    plt.close()
    print(f"    Saved: fig3_per_bit_ber.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 4: Training History
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_history(exp_dir):
    print("\n  [4/6] Training History...")

    hist_path = os.path.join(exp_dir, 'history.json')
    if not os.path.exists(hist_path):
        print(f"    Not found: {hist_path}")
        return

    with open(hist_path) as f:
        h = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Loss
    ax1.plot(h['steps'], h['train_loss'], color=STYLE['deeprx']['color'],
             linewidth=2, label='Training Loss')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss (BCE)')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # BER
    ax2.plot(h['steps'], h['train_ber'], color=STYLE['deeprx']['color'],
             linewidth=1, alpha=0.4, label='Train BER')

    if h['val_ber']:
        n_val = len(h['val_ber'])
        val_interval = h['steps'][-1] // n_val if n_val > 0 else 500
        val_steps = list(range(val_interval, val_interval *
                         n_val + 1, val_interval))[:n_val]

        ax2.plot(val_steps, h['val_ber'], color=STYLE['deeprx']['color'],
                 marker='o', linewidth=2.5, markersize=8, label='Val BER (DeepRx)')
        if h['val_ber_lmmse']:
            ax2.plot(val_steps[:len(h['val_ber_lmmse'])], h['val_ber_lmmse'],
                     color=STYLE['lmmse']['color'], marker='s', linewidth=2.5,
                     markersize=8, linestyle='--', label='Val BER (LMMSE)')

    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('BER')
    ax2.set_title('Bit Error Rate')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_training_history.pdf'))
    plt.close()
    print(f"    Saved: fig4_training_history.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 5: Channel Model Comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_channel_comparison(model, device='cpu'):
    print("\n  [5/6] Channel Model Comparison...")

    trad_rx = TraditionalReceiver('16QAM', device)
    channels = ['TDL_A', 'TDL_B', 'TDL_C', 'TDL_D', 'SIMPLE']
    labels = ['TDL-A\n(NLOS)', 'TDL-B\n(NLOS)', 'TDL-C\n(NLOS)',
              'TDL-D\n(LOS)', 'Simple\n(NLOS)']

    ber_deep, ber_lmmse, ci_d, ci_l = [], [], [], []

    for ch in channels:
        bd_trials, bl_trials = [], []
        for _ in range(20):
            data = generate_test_batch(8, 15.0, 100.0,
                                       channel_profile=ch, device=device)
            with torch.no_grad():
                logits = model(data['input'])
            bd_trials.append(compute_ber(logits, data['target_bits'],
                                         data['data_mask'], data['bit_mask']))
            llrs = trad_rx.process(data['rx_grid'], data['tx_pilots'],
                                   data['pilot_mask'])
            bl_trials.append(compute_ber(llrs, data['target_bits'],
                                         data['data_mask'], data['bit_mask']))

        bd_arr, bl_arr = np.array(bd_trials), np.array(bl_trials)
        ber_deep.append(bd_arr.mean())
        ber_lmmse.append(bl_arr.mean())
        ci_d.append(1.96 * bd_arr.std() / np.sqrt(20))
        ci_l.append(1.96 * bl_arr.std() / np.sqrt(20))

        gain = bl_arr.mean() / max(bd_arr.mean(), 1e-8)
        print(f"    {ch}: DeepRx={bd_arr.mean():.4f} | "
              f"LMMSE={bl_arr.mean():.4f} | Gain={gain:.2f}x")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(channels))
    width = 0.35

    ax.bar(x-width/2, ber_deep, width, yerr=ci_d,
           color=STYLE['deeprx']['color'], label='DeepRx', alpha=0.85, capsize=4)
    ax.bar(x+width/2, ber_lmmse, width, yerr=ci_l,
           color=STYLE['lmmse']['color'], label='LMMSE', alpha=0.85, capsize=4)

    for i in range(len(channels)):
        if ber_deep[i] > 0:
            gain = ber_lmmse[i] / ber_deep[i]
            color = 'green' if gain > 1 else 'red'
            ax.text(x[i], max(ber_deep[i], ber_lmmse[i]) + ci_l[i] + 0.001,
                    f'{gain:.2f}×', ha='center', fontsize=11,
                    color=color, fontweight='bold')

    ax.set_ylabel('BER')
    ax.set_title(
        'BER by Channel Model\n(SNR=15 dB, Doppler=100 Hz, 16-QAM, error bars=95% CI)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig5_channel_comparison.pdf'))
    plt.close()
    print(f"    Saved: fig5_channel_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 6: Gain Summary
# ═══════════════════════════════════════════════════════════════════════════

def plot_gain_summary(snr_results):
    print("\n  [6/6] Performance Gain Summary...")

    snr_list = snr_results['params']
    dm = np.array(snr_results['deeprx_mean'])
    lm = np.array(snr_results['lmmse_mean'])
    gains = lm / np.maximum(dm, 1e-8)

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ['#2E7D32' if g > 1 else '#C62828' for g in gains]
    bars = ax.bar(snr_list, gains, width=2.0, color=colors, alpha=0.75,
                  edgecolor='white', linewidth=1.5)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5,
               label='Equal Performance')
    ax.fill_between([snr_list[0]-2, snr_list[-1]+2], 1.0, max(gains)+0.5,
                    alpha=0.04, color='green')
    ax.fill_between([snr_list[0]-2, snr_list[-1]+2], 0, 1.0,
                    alpha=0.04, color='red')

    ax.text(snr_list[1], max(gains)*0.92, 'DeepRx Better ↑',
            fontsize=12, color='#2E7D32', fontweight='bold')
    ax.text(snr_list[1], 0.15, 'LMMSE Better ↓',
            fontsize=12, color='#C62828', fontweight='bold')

    for bar, g in zip(bars, gains):
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.02,
                f'{g:.2f}×', ha='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Gain (BER$_{LMMSE}$ / BER$_{DeepRx}$)')
    ax.set_title('DeepRx Performance Gain over LMMSE Baseline')
    ax.legend(loc='upper right')
    ax.set_xlim([snr_list[0]-2, snr_list[-1]+2])
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig6_gain_summary.pdf'))
    plt.close()
    print(f"    Saved: fig6_gain_summary.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print(f"{'DeepRx — Publication Quality Plots':^70}")
    print("=" * 70)

    setup_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    # Try publication model first, fall back to prototype
    for exp in ['deeprx_publication', 'deeprx_training_v1']:
        model_path = f'checkpoints/{exp}/best_model.pt'
        if os.path.exists(model_path):
            break

    print(f"  Loading: {model_path}")
    model = DeepRx(n_rx_antennas=2, max_bits_per_symbol=8)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Step: {ckpt['step']}, Best BER: {ckpt['best_val_ber']:.6f}")

    exp_dir = os.path.dirname(model_path)

    # Generate all plots
    snr_results = plot_ber_vs_snr(model, device)
    plot_ber_vs_doppler(model, device)
    plot_per_bit_ber(model, device)
    plot_training_history(exp_dir)
    plot_channel_comparison(model, device)
    plot_gain_summary(snr_results)

    print(f"\n{'='*70}")
    print(f"  All 6 figures saved to: {FIG_DIR}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
