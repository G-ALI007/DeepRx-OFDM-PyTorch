"""
============================================================================
DeepRx Step 2A: OFDM System — Modulation, Channel, Signal Generation
============================================================================
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional, Dict


# ═══════════════════════════════════════════════════════════════════════════
#  Section 1: QAM Modulation
# ═══════════════════════════════════════════════════════════════════════════

class QAMModulator:
    """
    QAM Modulator supporting QPSK, 16QAM, 64QAM, 256QAM.
    Gray-coded constellation normalized to unit average power.
    """

    CONSTELLATIONS = {}

    @staticmethod
    def _gray_code(n_bits: int) -> torch.Tensor:
        n = 2 ** n_bits
        codes = torch.zeros(n, dtype=torch.long)
        for i in range(n):
            codes[i] = i ^ (i >> 1)
        return codes

    @staticmethod
    def _build_constellation(modulation: str, device: str = 'cpu') -> Tuple[torch.Tensor, int]:
        config = {
            'QPSK':   (2, 2),
            '16QAM':  (4, 4),
            '64QAM':  (6, 8),
            '256QAM': (8, 16),
        }

        bits_per_symbol, M_per_dim = config[modulation]
        bits_per_dim = bits_per_symbol // 2

        gray = QAMModulator._gray_code(bits_per_dim)

        levels = torch.arange(M_per_dim, device=device, dtype=torch.float32)
        levels = 2 * levels - (M_per_dim - 1)

        pam = torch.zeros(M_per_dim, device=device)
        for i in range(M_per_dim):
            pam[gray[i]] = levels[i]

        M = M_per_dim ** 2
        symbols = torch.zeros(M, dtype=torch.cfloat, device=device)

        for i in range(M_per_dim):
            for q in range(M_per_dim):
                idx = i * M_per_dim + q
                symbols[idx] = torch.complex(pam[i], pam[q])

        avg_power = (symbols.abs() ** 2).mean()
        symbols = symbols / torch.sqrt(avg_power)

        return symbols, bits_per_symbol

    @staticmethod
    def get_constellation(modulation: str, device: str = 'cpu') -> Tuple[torch.Tensor, int]:
        key = (modulation, device)
        if key not in QAMModulator.CONSTELLATIONS:
            QAMModulator.CONSTELLATIONS[key] = QAMModulator._build_constellation(
                modulation, device
            )
        return QAMModulator.CONSTELLATIONS[key]

    @staticmethod
    def modulate(bits: torch.Tensor, modulation: str = '16QAM') -> torch.Tensor:
        device = bits.device
        constellation, bps = QAMModulator.get_constellation(
            modulation, str(device))

        powers = (2 ** torch.arange(bps - 1, -1, -1, device=device)).float()
        indices = (bits.float() * powers).sum(dim=-1).long()

        return constellation[indices]

    @staticmethod
    def bits_to_symbols(
        n_symbols: int,
        modulation: str = '16QAM',
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, bps = QAMModulator.get_constellation(modulation, device)
        bits = torch.randint(0, 2, (n_symbols, bps),
                             device=device, dtype=torch.float32)
        symbols = QAMModulator.modulate(bits, modulation)
        return bits, symbols


# ═══════════════════════════════════════════════════════════════════════════
#  Section 2: OFDM Transmitter
# ═══════════════════════════════════════════════════════════════════════════

class OFDMTransmitter:
    """
    OFDM Transmitter: Resource grid → Time-domain waveform.
    """

    def __init__(
        self,
        n_subcarriers: int = 312,
        n_fft: int = 512,
        cp_length: int = 36,
        n_symbols: int = 14
    ):
        self.F = n_subcarriers
        self.n_fft = n_fft
        self.cp_length = cp_length
        self.S = n_symbols

    def build_resource_grid(
        self,
        data_symbols: torch.Tensor,
        pilot_symbols: torch.Tensor,
        pilot_mask: torch.Tensor,
        data_bits: torch.Tensor,
        bits_per_symbol: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = data_symbols.shape[0]
        device = data_symbols.device

        data_mask = 1.0 - pilot_mask

        grid = pilot_symbols.clone()

        data_positions = data_mask.squeeze().bool()
        n_data_per_tti = int(data_positions.sum().item())

        for b in range(batch):
            grid[b, 0, data_positions] = data_symbols[b, :n_data_per_tti]

        B_max = 8
        target_bits = torch.zeros(batch, B_max, self.S, self.F, device=device)

        for b in range(batch):
            bits_flat = data_bits[b, :n_data_per_tti, :]
            if bits_per_symbol < B_max:
                pad = torch.zeros(n_data_per_tti, B_max -
                                  bits_per_symbol, device=device)
                bits_flat = torch.cat([bits_flat, pad], dim=-1)

            for bit_idx in range(B_max):
                target_bits[b, bit_idx, data_positions] = bits_flat[:, bit_idx]

        return grid, target_bits, data_mask

    def modulate_ofdm(self, grid: torch.Tensor) -> torch.Tensor:
        batch = grid.shape[0]
        device = grid.device

        fft_grid = torch.zeros(batch, self.S, self.n_fft,
                               dtype=torch.cfloat, device=device)

        start = (self.n_fft - self.F) // 2
        fft_grid[:, :, start:start + self.F] = grid[:, 0, :, :]

        time_symbols = torch.fft.ifft(fft_grid, dim=-1)

        cp = time_symbols[:, :, -self.cp_length:]
        ofdm_symbols = torch.cat([cp, time_symbols], dim=-1)

        tx_signal = ofdm_symbols.reshape(batch, -1)

        return tx_signal


# ═══════════════════════════════════════════════════════════════════════════
#  Section 3: Channel Models
# ═══════════════════════════════════════════════════════════════════════════

class ChannelModel:
    """
    Multipath fading channel with Doppler spread.
    """

    DELAY_PROFILES = {
        'TDL_A': {
            'delays': [0, 3, 5, 8, 11, 15, 18],
            'powers_db': [0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8],
            'is_los': False
        },
        'TDL_B': {
            'delays': [0, 1, 3, 5, 6, 9, 12],
            'powers_db': [0, -2.2, -0.6, -0.8, -4.0, -7.0, -11.0],
            'is_los': False
        },
        'TDL_C': {
            'delays': [0, 4, 6, 8, 14, 18, 22],
            'powers_db': [0, -4.4, -1.2, -4.0, -7.0, -12.0, -16.0],
            'is_los': False
        },
        'TDL_D': {
            'delays': [0, 1, 4, 8, 12, 16, 20],
            'powers_db': [-0.2, 0, -5.0, -8.0, -11.0, -14.0, -17.0],
            'is_los': True
        },
        'SIMPLE': {
            'delays': [0, 1, 2, 3, 4, 5, 6],
            'powers_db': [0, -2, -4, -6, -8, -10, -12],
            'is_los': False
        }
    }

    def __init__(
        self,
        profile: str = 'TDL_B',
        max_doppler_hz: float = 100.0,
        sampling_rate: float = 7.68e6,
        device: str = 'cpu'
    ):
        self.device = device
        self.max_doppler = max_doppler_hz
        self.fs = sampling_rate

        prof = self.DELAY_PROFILES[profile]
        self.delays = torch.tensor(prof['delays'], device=device)
        self.n_taps = len(self.delays)
        self.is_los = prof['is_los']

        powers_db = torch.tensor(
            prof['powers_db'], dtype=torch.float32, device=device)
        self.tap_powers = 10 ** (powers_db / 10)
        self.tap_powers = self.tap_powers / self.tap_powers.sum()

    def generate(
        self,
        batch: int,
        signal_length: int,
        n_ofdm_symbols: int = 14,
        n_fft: int = 512,
        cp_length: int = 36
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ofdm_sym_len = n_fft + cp_length

        h_taps = self._generate_doppler_taps(batch, n_ofdm_symbols)

        h_time = torch.zeros(batch, self.n_taps, signal_length,
                             dtype=torch.cfloat, device=self.device)

        for sym_idx in range(n_ofdm_symbols):
            start = sym_idx * ofdm_sym_len
            end = start + ofdm_sym_len
            if end > signal_length:
                end = signal_length

            for tap in range(self.n_taps):
                h_time[:, tap, start:end] = h_taps[:,
                                                   sym_idx, tap].unsqueeze(-1)

        h_freq = torch.zeros(batch, n_ofdm_symbols, n_fft,
                             dtype=torch.cfloat, device=self.device)

        for sym_idx in range(n_ofdm_symbols):
            for tap in range(self.n_taps):
                delay = self.delays[tap]
                freq_idx = torch.arange(
                    n_fft, device=self.device, dtype=torch.float32)
                phase = -2 * math.pi * freq_idx * delay / n_fft
                steering = torch.complex(torch.cos(phase), torch.sin(phase))
                h_freq[:, sym_idx, :] += (
                    h_taps[:, sym_idx,
                           tap].unsqueeze(-1) * steering.unsqueeze(0)
                )

        return h_time, h_freq

    def _generate_doppler_taps(
        self, batch: int, n_symbols: int
    ) -> torch.Tensor:
        ofdm_duration = 1e-3 / 14

        if self.max_doppler > 0:
            rho = float(np.real(
                np.exp(-2 * math.pi * self.max_doppler * ofdm_duration * 0.1)
            ))
            rho = max(min(rho, 0.9999), 0.0)
        else:
            rho = 1.0

        h_taps = torch.zeros(batch, n_symbols, self.n_taps,
                             dtype=torch.cfloat, device=self.device)

        for tap in range(self.n_taps):
            power = self.tap_powers[tap]
            std = torch.sqrt(power / 2)

            h_prev = std * torch.complex(
                torch.randn(batch, device=self.device),
                torch.randn(batch, device=self.device)
            )

            if self.is_los and tap == 0:
                k_factor = 10.0
                los_power = power * k_factor / (1 + k_factor)
                nlos_power = power / (1 + k_factor)
                std = torch.sqrt(nlos_power / 2)
                h_prev = std * torch.complex(
                    torch.randn(batch, device=self.device),
                    torch.randn(batch, device=self.device)
                ) + torch.sqrt(torch.tensor(los_power, device=self.device))

            h_taps[:, 0, tap] = h_prev

            for sym in range(1, n_symbols):
                innovation = std * torch.complex(
                    torch.randn(batch, device=self.device),
                    torch.randn(batch, device=self.device)
                )
                h_prev = rho * h_prev + math.sqrt(1 - rho ** 2) * innovation
                h_taps[:, sym, tap] = h_prev

        return h_taps

    def apply_channel(
        self,
        tx_signal: torch.Tensor,
        h_time: torch.Tensor
    ) -> torch.Tensor:
        batch, sig_len = tx_signal.shape
        rx_signal = torch.zeros_like(tx_signal)

        for tap in range(self.n_taps):
            delay = int(self.delays[tap].item())
            if delay == 0:
                rx_signal += h_time[:, tap, :] * tx_signal
            else:
                rx_signal[:, delay:] += (
                    h_time[:, tap, delay:] * tx_signal[:, :sig_len - delay]
                )

        return rx_signal


# ═══════════════════════════════════════════════════════════════════════════
#  Section 4: OFDM Receiver Front-End
# ═══════════════════════════════════════════════════════════════════════════

class OFDMReceiver:
    """OFDM Receiver front-end: Time-domain → Frequency-domain."""

    def __init__(
        self,
        n_subcarriers: int = 312,
        n_fft: int = 512,
        cp_length: int = 36,
        n_symbols: int = 14
    ):
        self.F = n_subcarriers
        self.n_fft = n_fft
        self.cp_length = cp_length
        self.S = n_symbols

    def demodulate(
        self,
        rx_waveform: torch.Tensor,
        n_rx: int = 1
    ) -> torch.Tensor:
        if rx_waveform.dim() == 2:
            rx_waveform = rx_waveform.unsqueeze(1)

        batch = rx_waveform.shape[0]
        device = rx_waveform.device
        ofdm_sym_len = self.n_fft + self.cp_length

        rx_grid = torch.zeros(batch, n_rx, self.S, self.F,
                              dtype=torch.cfloat, device=device)

        start_sc = (self.n_fft - self.F) // 2

        for sym_idx in range(self.S):
            start = sym_idx * ofdm_sym_len + self.cp_length
            end = start + self.n_fft

            symbol_data = rx_waveform[:, :, start:end]
            freq_data = torch.fft.fft(symbol_data, dim=-1)
            rx_grid[:, :, sym_idx, :] = freq_data[:,
                                                  :, start_sc:start_sc + self.F]

        return rx_grid


# ═══════════════════════════════════════════════════════════════════════════
#  Section 5: Noise & Interference
# ═══════════════════════════════════════════════════════════════════════════

def add_awgn(
    signal: torch.Tensor,
    snr_db: float,
    signal_power: Optional[float] = None
) -> torch.Tensor:
    if signal_power is None:
        signal_power = (signal.abs() ** 2).mean().item()

    noise_power = signal_power / (10 ** (snr_db / 10))
    std = math.sqrt(noise_power / 2)

    noise = std * torch.complex(
        torch.randn_like(signal.real),
        torch.randn_like(signal.imag)
    )

    return signal + noise


def generate_interference(
    batch: int,
    signal_length: int,
    sir_db: float,
    signal_power: float,
    channel_model: ChannelModel,
    n_fft: int = 512,
    cp_length: int = 36,
    n_symbols: int = 14,
    device: str = 'cpu'
) -> torch.Tensor:
    ofdm_sym_len = n_fft + cp_length

    intf_grid = (1.0 / math.sqrt(2)) * torch.complex(
        torch.randn(batch, n_symbols, n_fft, device=device),
        torch.randn(batch, n_symbols, n_fft, device=device)
    )

    time_symbols = torch.fft.ifft(intf_grid, dim=-1)
    cp = time_symbols[:, :, -cp_length:]
    ofdm_symbols = torch.cat([cp, time_symbols], dim=-1)
    intf_signal = ofdm_symbols.reshape(batch, -1)

    if intf_signal.shape[1] < signal_length:
        pad = torch.zeros(batch, signal_length - intf_signal.shape[1],
                          dtype=torch.cfloat, device=device)
        intf_signal = torch.cat([intf_signal, pad], dim=1)
    else:
        intf_signal = intf_signal[:, :signal_length]

    h_time_intf, _ = channel_model.generate(
        batch, signal_length, n_symbols, n_fft, cp_length
    )
    intf_signal = channel_model.apply_channel(intf_signal, h_time_intf)

    intf_power = (intf_signal.abs() ** 2).mean().item()
    desired_power = signal_power / (10 ** (sir_db / 10))
    scale = math.sqrt(desired_power / max(intf_power, 1e-10))

    return intf_signal * scale


# ═══════════════════════════════════════════════════════════════════════════
#  Verification
# ═══════════════════════════════════════════════════════════════════════════

def verify_ofdm_system():
    print("\n" + "=" * 70)
    print(f"{'OFDM System Verification':^70}")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch = 4
    S, F = 14, 312
    n_fft, cp_len = 512, 36
    modulation = '16QAM'

    # Test 1: QAM
    print(f"\n{'─'*50}")
    print("  Test 1: QAM Modulation")

    for mod in ['QPSK', '16QAM', '64QAM', '256QAM']:
        const, bps = QAMModulator.get_constellation(mod, device)
        avg_pow = (const.abs() ** 2).mean().item()
        print(
            f"    {mod:>7}: {len(const):>3} symbols, {bps} bps, avg power = {avg_pow:.4f}")
        assert abs(avg_pow - 1.0) < 0.01

    bits, symbols = QAMModulator.bits_to_symbols(1000, '16QAM', device)
    assert bits.shape == (1000, 4)
    assert symbols.shape == (1000,)
    print("    ✓ Passed")

    # Test 2: OFDM TX
    print(f"\n{'─'*50}")
    print("  Test 2: OFDM Transmitter")

    tx = OFDMTransmitter(F, n_fft, cp_len, S)

    from deeprx_model import create_pilot_mask, generate_qpsk_pilots

    pilot_mask = create_pilot_mask(S, F, '2_pilots_A', device)
    pilot_symbols = generate_qpsk_pilots(batch, S, F, pilot_mask, device)
    data_mask = 1.0 - pilot_mask
    n_data = int(data_mask.sum().item())

    data_bits, data_syms = QAMModulator.bits_to_symbols(
        batch * n_data, modulation, device
    )
    data_syms = data_syms.reshape(batch, n_data)
    data_bits = data_bits.reshape(batch, n_data, -1)

    grid, target_bits, dmask = tx.build_resource_grid(
        data_syms, pilot_symbols, pilot_mask, data_bits, 4
    )

    tx_signal = tx.modulate_ofdm(grid)
    expected_len = S * (n_fft + cp_len)

    print(f"    Grid shape:    {grid.shape}")
    print(f"    TX signal:     {tx_signal.shape}")
    assert tx_signal.shape == (batch, expected_len)
    print("    ✓ Passed")

    # Test 3: Channel
    print(f"\n{'─'*50}")
    print("  Test 3: Channel Model")

    sig_len = tx_signal.shape[1]

    for profile in ['TDL_A', 'TDL_B', 'TDL_D', 'SIMPLE']:
        ch = ChannelModel(profile, max_doppler_hz=200.0, device=device)
        h_t, h_f = ch.generate(batch, sig_len, S, n_fft, cp_len)
        rx = ch.apply_channel(tx_signal, h_t)
        print(
            f"    {profile}: h_time={h_t.shape}, h_freq={h_f.shape}, LOS={ch.is_los}")
        assert rx.shape == tx_signal.shape

    print("    ✓ Passed")

    # Test 4: OFDM RX
    print(f"\n{'─'*50}")
    print("  Test 4: OFDM Receiver")

    ch = ChannelModel('TDL_B', max_doppler_hz=100.0, device=device)
    h_t, h_f = ch.generate(batch, sig_len, S, n_fft, cp_len)
    rx_waveform = ch.apply_channel(tx_signal, h_t)
    rx_waveform = add_awgn(rx_waveform, snr_db=20.0)

    rx_frontend = OFDMReceiver(F, n_fft, cp_len, S)
    rx_grid = rx_frontend.demodulate(rx_waveform, n_rx=1)

    print(f"    RX grid: {rx_grid.shape}")
    assert rx_grid.shape == (batch, 1, S, F)
    print("    ✓ Passed")

    # Test 5: Noise
    print(f"\n{'─'*50}")
    print("  Test 5: Noise & Interference")

    sig_pow = (tx_signal.abs() ** 2).mean().item()

    for snr in [0, 10, 20, 30]:
        noisy = add_awgn(tx_signal, snr, sig_pow)
        noise = noisy - tx_signal
        actual_snr = 10 * math.log10(
            sig_pow / max((noise.abs() ** 2).mean().item(), 1e-10)
        )
        print(f"    Target SNR={snr:>3} dB, Actual={actual_snr:>6.1f} dB")

    print("    ✓ Passed")

    # Test 6: Pipeline
    print(f"\n{'─'*50}")
    print("  Test 6: Full TX → RX Pipeline")

    rx_clean = rx_frontend.demodulate(tx_signal, n_rx=1)
    pilot_pos = pilot_mask.squeeze().bool()
    tx_at_pilots = grid[0, 0, pilot_pos]
    rx_at_pilots = rx_clean[0, 0, pilot_pos]

    error = (tx_at_pilots - rx_at_pilots).abs().mean().item()
    print(f"    Recovery error: {error:.6f}")
    assert error < 1e-4
    print("    ✓ Passed")

    print(f"\n{'='*70}")
    print(f"{'ALL TESTS PASSED':^70}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    verify_ofdm_system()
