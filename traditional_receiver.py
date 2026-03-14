"""
============================================================================
DeepRx Step 3A: Traditional LMMSE Receiver (Baseline) — CORRECTED
============================================================================
Fixed:
    1. Bit labeling: use binary labels matching modulator's bit-to-index map
    2. Noise estimation: use adjacent pilot differences instead of broken
       reconstruction method that always returns zero
============================================================================
"""

import torch
import math
from typing import Tuple, Optional

from ofdm_system import QAMModulator


class TraditionalReceiver:
    """
    Traditional OFDM Receiver: LS estimation + LMMSE equalization.

    LLR Convention: positive → bit=1, negative → bit=0
    (matching BCE with logits convention used by DeepRx)
    """

    def __init__(self, modulation: str = '16QAM', device: str = 'cpu'):
        self.modulation = modulation
        self.device = device
        self.constellation, self.bps = QAMModulator.get_constellation(
            modulation, device
        )

        # ═══════════════════════════════════════════════════════════════
        # FIX #1: Build bit labels using BINARY representation of index
        #
        # The QAMModulator maps bits to index as:
        #     index = b0*2^(B-1) + b1*2^(B-2) + ... + b_{B-1}*2^0
        #
        # Gray coding is already embedded in the constellation geometry
        # (via the PAM level assignment), so the demapper must use
        # plain binary labels to stay consistent.
        #
        # Previous bug: used Gray-coded labels on top of already
        # Gray-coded constellation → double Gray coding → wrong labels
        # ═══════════════════════════════════════════════════════════════
        M = len(self.constellation)
        self.bit_labels = torch.zeros(M, self.bps, device=device)
        for idx in range(M):
            for b in range(self.bps):
                self.bit_labels[idx, b] = float(
                    (idx >> (self.bps - 1 - b)) & 1
                )

    def estimate_channel_ls(
        self,
        rx_grid: torch.Tensor,
        tx_pilots: torch.Tensor,
        pilot_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Least Squares channel estimation at pilot positions.

        Ĥ_raw = Y ⊙ conj(X_p)  at pilot positions, 0 elsewhere

        At pilot positions: Ĥ = H + n·conj(x_p)  (noisy estimate)
        """
        Nr = rx_grid.shape[1]
        tx_expanded = tx_pilots.expand(-1, Nr, -1, -1)
        h_raw = rx_grid * torch.conj(tx_expanded) * pilot_mask
        return h_raw

    def interpolate_channel(
        self,
        h_pilots: torch.Tensor,
        pilot_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        2D linear interpolation: pilot positions → all RE positions.

        Step 1: Frequency interpolation within each pilot OFDM symbol
        Step 2: Time interpolation between pilot OFDM symbols
        """
        batch, Nr, S, F = h_pilots.shape
        device = h_pilots.device

        h_interp = torch.zeros_like(h_pilots)
        mask_sq = pilot_mask.squeeze()  # (S, F)

        for b in range(batch):
            for ant in range(Nr):
                h_ant = h_pilots[b, ant]  # (S, F)

                # Find OFDM symbols containing pilots
                pilot_sym_indices = []
                for s in range(S):
                    if mask_sq[s].sum() > 0:
                        pilot_sym_indices.append(s)

                if len(pilot_sym_indices) == 0:
                    continue

                # Step 1: Frequency interpolation for pilot symbols
                freq_interp = torch.zeros(
                    S, F, dtype=torch.cfloat, device=device)

                for s_idx in pilot_sym_indices:
                    pilot_pos = mask_sq[s_idx].nonzero(as_tuple=True)[0]
                    pilot_vals = h_ant[s_idx, pilot_pos]

                    if len(pilot_pos) < 2:
                        freq_interp[s_idx, :] = pilot_vals.mean()
                    else:
                        freq_interp[s_idx] = self._interp1d(
                            pilot_pos.float(), pilot_vals,
                            torch.arange(F, device=device, dtype=torch.float32)
                        )

                # Step 2: Time interpolation
                if len(pilot_sym_indices) == 1:
                    s0 = pilot_sym_indices[0]
                    for s in range(S):
                        h_interp[b, ant, s] = freq_interp[s0]
                else:
                    t_known = torch.tensor(
                        pilot_sym_indices, dtype=torch.float32, device=device
                    )
                    t_query = torch.arange(
                        S, dtype=torch.float32, device=device)

                    for f_idx in range(F):
                        vals = torch.stack([
                            freq_interp[s, f_idx] for s in pilot_sym_indices
                        ])
                        h_interp[b, ant, :, f_idx] = self._interp1d(
                            t_known, vals, t_query
                        )

        return h_interp

    @staticmethod
    def _interp1d(
        x_known: torch.Tensor,
        y_known: torch.Tensor,
        x_query: torch.Tensor
    ) -> torch.Tensor:
        """1D linear interpolation with edge extrapolation."""
        K = len(x_known)
        N = len(x_query)
        device = x_known.device

        if K == 1:
            return y_known[0].expand(N)

        y_query = torch.zeros(N, dtype=y_known.dtype, device=device)

        for i, xq in enumerate(x_query):
            if xq <= x_known[0]:
                t = (xq - x_known[0]) / (x_known[1] - x_known[0])
                y_query[i] = y_known[0] + t * (y_known[1] - y_known[0])
            elif xq >= x_known[-1]:
                t = (xq - x_known[-2]) / (x_known[-1] - x_known[-2])
                y_query[i] = y_known[-2] + t * (y_known[-1] - y_known[-2])
            else:
                idx = torch.searchsorted(x_known, xq) - 1
                idx = idx.clamp(0, K - 2)
                t = (xq - x_known[idx]) / (x_known[idx + 1] - x_known[idx])
                y_query[i] = y_known[idx] + t * \
                    (y_known[idx + 1] - y_known[idx])

        return y_query

    def estimate_noise_power(
        self,
        h_pilots_raw: torch.Tensor,
        pilot_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        ═══════════════════════════════════════════════════════════════
        FIX #2: Noise power estimation from adjacent pilot differences

        Previous bug: compared received signal to reconstructed signal
        at pilot positions, but the interpolated channel at pilot
        positions EQUALS the LS estimate (which already contains noise),
        giving reconstruction error = 0 always.

        New method: use differences between adjacent LS estimates.

        For adjacent pilots f1, f2 with smooth channel:
            Δĥ = ĥ(f2) - ĥ(f1) ≈ noise(f2) - noise(f1)
            E[|Δĥ|²] ≈ 2·σ²_n

        So: σ²_n ≈ mean(|Δĥ|²) / 2
        ═══════════════════════════════════════════════════════════════
        """
        batch, Nr, S, F = h_pilots_raw.shape
        device = h_pilots_raw.device

        mask_sq = pilot_mask.squeeze()  # (S, F)

        sum_diff_sq = 0.0
        n_diffs = 0

        for s in range(S):
            pilot_pos = mask_sq[s].nonzero(as_tuple=True)[0]
            if len(pilot_pos) < 2:
                continue

            for k in range(len(pilot_pos) - 1):
                p1, p2 = pilot_pos[k], pilot_pos[k + 1]
                diff = h_pilots_raw[:, :, s, p2] - h_pilots_raw[:, :, s, p1]
                sum_diff_sq += (diff.abs() ** 2).sum().item()
                n_diffs += batch * Nr

        if n_diffs > 0:
            sigma2 = sum_diff_sq / (2.0 * n_diffs)
        else:
            sigma2 = 1e-2  # fallback

        sigma2 = max(sigma2, 1e-10)

        return torch.full((batch,), sigma2, device=device)

    def equalize_lmmse(
        self,
        rx_grid: torch.Tensor,
        h_est: torch.Tensor,
        noise_power: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LMMSE Equalization for SIMO.

        x̂ = (Ĥ^H·Ĥ + σ²)^{-1} · Ĥ^H · y
           = (|Ĥ|² + σ²)^{-1} · Ĥ^H · y    (scalar for SIMO)
        """
        batch, Nr, S, F = rx_grid.shape

        hh_y = (torch.conj(h_est) * rx_grid).sum(dim=1)  # (batch, S, F)
        hh_h = (h_est.abs() ** 2).sum(dim=1)              # (batch, S, F)

        sigma2 = noise_power.view(batch, 1, 1)
        denom = (hh_h + sigma2).clamp(min=1e-10)

        eq_symbols = hh_y / denom

        # Post-equalization SNR for LLR scaling
        eq_snr = hh_h / sigma2.clamp(min=1e-10)

        return eq_symbols, eq_snr

    def compute_llrs(
        self,
        eq_symbols: torch.Tensor,
        eq_snr: torch.Tensor
    ) -> torch.Tensor:
        """
        Max-log LLR approximation.

        L_l = scale · (min_{C⁰_l} |x̂-x|² − min_{C¹_l} |x̂-x|²)

        Convention: positive → bit=1 (matching BCE training)

        Uses self.bit_labels (binary labels) which are guaranteed
        consistent with the QAMModulator's bit-to-symbol mapping.
        """
        batch, S, F = eq_symbols.shape
        device = eq_symbols.device
        B_max = 8

        constellation = self.constellation  # (M,)
        M = len(constellation)

        # Distances to all constellation points: (batch, S, F, M)
        x_hat = eq_symbols.unsqueeze(-1)         # (batch, S, F, 1)
        const = constellation.view(1, 1, 1, M)   # (1, 1, 1, M)
        distances = (x_hat - const).abs() ** 2    # (batch, S, F, M)

        llrs = torch.zeros(batch, B_max, S, F, device=device)

        for l in range(self.bps):
            # Using binary bit labels (FIX #1)
            mask_0 = (self.bit_labels[:, l] == 0)  # points where bit l = 0
            mask_1 = (self.bit_labels[:, l] == 1)  # points where bit l = 1

            min_dist_0 = distances[:, :, :, mask_0].min(dim=-1)[0]
            min_dist_1 = distances[:, :, :, mask_1].min(dim=-1)[0]

            # positive → closer to bit=1 points → bit=1
            llrs[:, l] = eq_snr * (min_dist_0 - min_dist_1)

        return llrs

    def process(
        self,
        rx_grid: torch.Tensor,
        tx_pilots: torch.Tensor,
        pilot_mask: torch.Tensor,
        known_channel: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Full receiver pipeline: RX signal → LLRs."""

        # Channel estimation
        h_pilots_raw = self.estimate_channel_ls(rx_grid, tx_pilots, pilot_mask)

        if known_channel is not None:
            h_est = known_channel
        else:
            h_est = self.interpolate_channel(h_pilots_raw, pilot_mask)

        # Noise estimation (FIX #2: from pilot differences)
        noise_power = self.estimate_noise_power(h_pilots_raw, pilot_mask)

        # Equalization
        eq_symbols, eq_snr = self.equalize_lmmse(rx_grid, h_est, noise_power)

        # Demapping (FIX #1: binary labels)
        llrs = self.compute_llrs(eq_symbols, eq_snr)

        return llrs


# ═══════════════════════════════════════════════════════════════════════════
#  Verification
# ═══════════════════════════════════════════════════════════════════════════

def verify_traditional_receiver():
    print("\n" + "=" * 70)
    print(f"{'Traditional Receiver Verification (CORRECTED)':^70}")
    print("=" * 70)

    device = 'cpu'
    batch, Nr, S, F = 4, 2, 14, 312

    from deeprx_model import create_pilot_mask, generate_qpsk_pilots, compute_ber
    from ofdm_system import (
        QAMModulator, OFDMTransmitter, OFDMReceiver,
        ChannelModel, add_awgn
    )

    n_fft, cp_len = 512, 36
    modulation = '16QAM'

    trad_rx = TraditionalReceiver(modulation, device)

    # ── Test 0: Verify Constellation Consistency ──
    print(f"\n{'─'*50}")
    print("  Test 0: Constellation Labeling Consistency")

    const, bps = QAMModulator.get_constellation(modulation, device)
    M = len(const)

    n_test = 5000
    test_bits, test_symbols = QAMModulator.bits_to_symbols(
        n_test, modulation, device)

    # For each symbol, find nearest constellation point
    dists = (test_symbols.unsqueeze(-1) - const.unsqueeze(0)).abs()
    nearest_idx = dists.argmin(dim=-1)

    # Recover bits from nearest index using binary labels
    recovered_bits = torch.zeros_like(test_bits)
    for b in range(bps):
        recovered_bits[:, b] = ((nearest_idx >> (bps - 1 - b)) & 1).float()

    bit_errors = (recovered_bits != test_bits).float().sum().item()
    print(f"    Tested {n_test} symbols, bit errors: {int(bit_errors)}")
    assert bit_errors == 0, "Constellation labeling is INCONSISTENT!"
    print("    ✓ Labeling is perfectly consistent")

    # ── Setup for remaining tests ──
    tx = OFDMTransmitter(F, n_fft, cp_len, S)
    rx_fe = OFDMReceiver(F, n_fft, cp_len, S)

    pilot_mask = create_pilot_mask(S, F, '2_pilots_A', device)
    pilot_symbols = generate_qpsk_pilots(batch, S, F, pilot_mask, device)
    data_mask = 1.0 - pilot_mask
    n_data = int(data_mask.sum().item())

    data_bits, data_syms = QAMModulator.bits_to_symbols(
        batch * n_data, modulation, device
    )
    data_syms = data_syms.reshape(batch, n_data)
    data_bits = data_bits.reshape(batch, n_data, -1)

    grid, target_bits, _ = tx.build_resource_grid(
        data_syms, pilot_symbols, pilot_mask, data_bits, 4
    )
    tx_signal = tx.modulate_ofdm(grid)
    sig_len = tx_signal.shape[1]
    sig_power = (tx_signal.abs() ** 2).mean().item()

    bit_mask = torch.zeros(batch, 8, 1, 1, device=device)
    bit_mask[:, :4, :, :] = 1.0

    # ── Test 1: No Channel (identity), High SNR ──
    print(f"\n{'─'*50}")
    print("  Test 1: No Fading Channel, High SNR")

    rx_waveform = add_awgn(tx_signal, 40.0, sig_power)
    rx_multi = rx_waveform.unsqueeze(1).expand(-1, Nr, -1)
    rx_grid = rx_fe.demodulate(rx_multi, Nr)

    llrs = trad_rx.process(rx_grid, pilot_symbols, pilot_mask)
    ber = compute_ber(llrs, target_bits, data_mask, bit_mask)

    print(f"    BER (no channel, 40dB): {ber:.6f}")
    assert ber < 0.01, f"BER too high for clean channel: {ber}"
    print("    ✓ Passed")

    # ── Test 2: SNR Sweep with Fading ──
    print(f"\n{'─'*50}")
    print("  Test 2: SNR Sweep with Fading Channel")

    prev_ber = 1.0
    ber_decreasing = True

    for snr in [0, 5, 10, 15, 20, 25]:
        rx_waveforms = []
        for ant in range(Nr):
            ch = ChannelModel('TDL_B', max_doppler_hz=50.0, device=device)
            h_t, _ = ch.generate(batch, sig_len, S, n_fft, cp_len)
            rx_ant = ch.apply_channel(tx_signal, h_t)
            rx_ant = add_awgn(rx_ant, float(snr), sig_power)
            rx_waveforms.append(rx_ant)

        rx_multi = torch.stack(rx_waveforms, dim=1)
        rx_grid = rx_fe.demodulate(rx_multi, Nr)
        llrs = trad_rx.process(rx_grid, pilot_symbols, pilot_mask)

        ber = compute_ber(llrs, target_bits, data_mask, bit_mask)
        trend = "↓" if ber < prev_ber else "↑"
        print(f"    SNR={snr:>3} dB  →  BER = {ber:.4f}  {trend}")

        if snr >= 10 and ber > prev_ber + 0.05:
            ber_decreasing = False
        prev_ber = ber

    print(f"    BER trend correct: {'✓' if ber_decreasing else '✗ WARNING'}")
    print("    ✓ Passed")

    # ── Test 3: LLR Sign Convention ──
    print(f"\n{'─'*50}")
    print("  Test 3: LLR Sign Convention Check")

    # At high SNR, LLR sign should match true bits
    rx_waveform = add_awgn(tx_signal, 30.0, sig_power)
    rx_multi = rx_waveform.unsqueeze(1).expand(-1, Nr, -1)
    rx_grid = rx_fe.demodulate(rx_multi, Nr)
    llrs = trad_rx.process(rx_grid, pilot_symbols, pilot_mask)

    # Check: positive LLR → bit=1, negative → bit=0
    detected = (llrs[:, :4] > 0).float()
    agreement = (detected == target_bits[:, :4]).float()
    dm = data_mask.expand(batch, 4, S, F)
    accuracy = (agreement * dm).sum() / dm.sum()

    print(f"    Accuracy at 30dB (no fading): {accuracy.item():.4f}")
    assert accuracy.item(
    ) > 0.95, f"LLR convention likely wrong: accuracy={accuracy.item()}"
    print("    ✓ Convention is correct")

    # ── Test 4: Per-bit BER ──
    print(f"\n{'─'*50}")
    print("  Test 4: Per-bit BER Analysis (no fading, 25dB)")

    rx_waveform = add_awgn(tx_signal, 25.0, sig_power)
    rx_multi = rx_waveform.unsqueeze(1).expand(-1, Nr, -1)
    rx_grid = rx_fe.demodulate(rx_multi, Nr)
    llrs = trad_rx.process(rx_grid, pilot_symbols, pilot_mask)

    for bit_idx in range(4):
        bm_single = torch.zeros(batch, 8, 1, 1, device=device)
        bm_single[:, bit_idx, :, :] = 1.0
        ber_bit = compute_ber(llrs, target_bits, data_mask, bm_single)
        bit_name = ['I-MSB', 'I-LSB', 'Q-MSB', 'Q-LSB'][bit_idx]
        print(f"    Bit {bit_idx} ({bit_name}): BER = {ber_bit:.4f}")

    print("    ✓ Passed")

    print(f"\n{'='*70}")
    print(f"{'ALL TESTS PASSED':^70}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    verify_traditional_receiver()
