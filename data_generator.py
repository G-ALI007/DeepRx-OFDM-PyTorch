"""
============================================================================
DeepRx Step 2B: Training Data Generator
============================================================================
"""

import torch
from torch.utils.data import Dataset, DataLoader
import math
import random
from typing import Dict, Optional, Tuple

from deeprx_model import (
    create_pilot_mask, generate_qpsk_pilots,
    build_deeprx_input, create_bit_mask, MODULATION_CONFIG
)
from ofdm_system import (
    QAMModulator, OFDMTransmitter, ChannelModel,
    OFDMReceiver, add_awgn, generate_interference
)


class DeepRxDataset(Dataset):
    """
    On-the-fly training data generator for DeepRx.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        n_rx_antennas: int = 2,
        n_subcarriers: int = 312,
        n_fft: int = 512,
        cp_length: int = 36,
        n_ofdm_symbols: int = 14,
        modulation: str = '16QAM',
        snr_range: Tuple[float, float] = (-4.0, 32.0),
        doppler_range: Tuple[float, float] = (0.0, 500.0),
        channel_profiles: Optional[list] = None,
        pilot_configs: Optional[list] = None,
        add_interference: bool = False,
        sir_range: Tuple[float, float] = (0.0, 36.0),
        device: str = 'cpu'
    ):
        self.n_samples = n_samples
        self.Nr = n_rx_antennas
        self.F = n_subcarriers
        self.n_fft = n_fft
        self.cp_length = cp_length
        self.S = n_ofdm_symbols
        self.modulation = modulation
        self.snr_range = snr_range
        self.doppler_range = doppler_range
        self.add_interference = add_interference
        self.sir_range = sir_range
        self.device = device

        self.bps = MODULATION_CONFIG[modulation]

        if channel_profiles is None:
            channel_profiles = ['TDL_A', 'TDL_B', 'TDL_C', 'TDL_D', 'SIMPLE']
        self.channel_profiles = channel_profiles

        if pilot_configs is None:
            pilot_configs = ['1_pilot_A', '1_pilot_B',
                             '2_pilots_A', '2_pilots_B']
        self.pilot_configs = pilot_configs

        self.tx = OFDMTransmitter(self.F, self.n_fft, self.cp_length, self.S)
        self.rx = OFDMReceiver(self.F, self.n_fft, self.cp_length, self.S)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        device = self.device
        B_max = 8

        snr_db = random.uniform(*self.snr_range)
        doppler_hz = random.uniform(*self.doppler_range)
        profile = random.choice(self.channel_profiles)
        pilot_cfg = random.choice(self.pilot_configs)

        pilot_mask = create_pilot_mask(self.S, self.F, pilot_cfg, device)
        data_mask = 1.0 - pilot_mask
        bit_mask = create_bit_mask(self.modulation, B_max, device)
        n_data = int(data_mask.sum().item())

        data_bits, data_syms = QAMModulator.bits_to_symbols(
            n_data, self.modulation, device
        )
        data_syms = data_syms.unsqueeze(0)
        data_bits = data_bits.unsqueeze(0)

        pilot_symbols = generate_qpsk_pilots(
            1, self.S, self.F, pilot_mask, device)

        grid, target_bits, _ = self.tx.build_resource_grid(
            data_syms, pilot_symbols, pilot_mask, data_bits, self.bps
        )

        tx_signal = self.tx.modulate_ofdm(grid)
        sig_len = tx_signal.shape[1]
        sig_power = (tx_signal.abs() ** 2).mean().item()

        rx_waveforms = []

        for ant in range(self.Nr):
            ch = ChannelModel(profile, doppler_hz, device=device)
            h_time, h_freq = ch.generate(
                1, sig_len, self.S, self.n_fft, self.cp_length)

            rx_ant = ch.apply_channel(tx_signal, h_time)
            rx_ant = add_awgn(rx_ant, snr_db, sig_power)

            if self.add_interference:
                sir_db = random.uniform(*self.sir_range)
                intf = generate_interference(
                    1, sig_len, sir_db, sig_power, ch,
                    self.n_fft, self.cp_length, self.S, device
                )
                rx_ant = rx_ant + intf

            rx_waveforms.append(rx_ant)

        rx_multi = torch.cat(rx_waveforms, dim=0).unsqueeze(0)

        rx_grid = self.rx.demodulate(rx_multi, self.Nr)

        deeprx_input = build_deeprx_input(rx_grid, pilot_symbols)

        return {
            'input':       deeprx_input.squeeze(0),
            'target_bits': target_bits.squeeze(0),
            'data_mask':   data_mask.squeeze(0),
            'bit_mask':    bit_mask.squeeze(0).squeeze(-1).squeeze(-1),
            'snr_db':      torch.tensor(snr_db),
            'doppler_hz':  torch.tensor(doppler_hz),
        }


def verify_data_generator():
    print("\n" + "=" * 70)
    print(f"{'Data Generator Verification':^70}")
    print("=" * 70)

    device = 'cpu'
    Nr = 2
    S, F = 14, 312
    B_max = 8
    Nc = 2 * Nr + 1

    # Test 1
    print(f"\n{'─'*50}")
    print("  Test 1: Single Sample Generation")

    dataset = DeepRxDataset(
        n_samples=100,
        n_rx_antennas=Nr,
        modulation='16QAM',
        snr_range=(5.0, 20.0),
        doppler_range=(10.0, 200.0),
        add_interference=False,
        device=device
    )

    sample = dataset[0]

    print(f"    input:       {sample['input'].shape}")
    print(f"    target_bits: {sample['target_bits'].shape}")
    print(f"    data_mask:   {sample['data_mask'].shape}")
    print(f"    bit_mask:    {sample['bit_mask'].shape}")
    print(f"    SNR:         {sample['snr_db'].item():.1f} dB")
    print(f"    Doppler:     {sample['doppler_hz'].item():.1f} Hz")

    assert sample['input'].shape == (2 * Nc, S, F)
    assert sample['target_bits'].shape == (B_max, S, F)
    print("    ✓ Passed")

    # Test 2
    print(f"\n{'─'*50}")
    print("  Test 2: DataLoader Integration")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    print(f"    Batch input:       {batch['input'].shape}")
    print(f"    Batch target_bits: {batch['target_bits'].shape}")
    assert batch['input'].shape[0] == 4
    print("    ✓ Passed")

    # Test 3
    print(f"\n{'─'*50}")
    print("  Test 3: Compatibility with DeepRx Model")

    from deeprx_model import DeepRx, DeepRxLoss, compute_ber

    model = DeepRx(n_rx_antennas=Nr, max_bits_per_symbol=B_max)
    criterion = DeepRxLoss()

    model.eval()
    with torch.no_grad():
        logits = model(batch['input'])
        bit_mask_4d = batch['bit_mask'].unsqueeze(-1).unsqueeze(-1)
        loss = criterion(logits, batch['target_bits'],
                         batch['data_mask'], bit_mask_4d)
        ber = compute_ber(
            logits, batch['target_bits'], batch['data_mask'], bit_mask_4d)

    print(f"    Output: {logits.shape}")
    print(f"    Loss:   {loss.item():.4f}")
    print(f"    BER:    {ber:.4f}")
    assert logits.shape == (4, B_max, S, F)
    print("    ✓ Passed")

    print(f"\n{'='*70}")
    print(f"{'ALL TESTS PASSED':^70}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    verify_data_generator()
