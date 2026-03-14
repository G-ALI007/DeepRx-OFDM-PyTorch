"""
DeepRx: Model Architecture, Input Construction, Loss Function — FINAL CORRECTED
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional, Dict


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3),
                 dilation=(1, 1), depth_multiplier=2, bias=False):
        super().__init__()
        mid_channels = in_channels * depth_multiplier
        padding = (dilation[0]*(kernel_size[0]-1)//2,
                   dilation[1]*(kernel_size[1]-1)//2)
        self.depthwise = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size,
                                   padding=padding, dilation=dilation,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class PreactivationResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3),
                 dilation=(1, 1), depth_multiplier=2):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.dsconv1 = DepthwiseSeparableConv2d(in_channels, out_channels,
                                                kernel_size, dilation, depth_multiplier)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dsconv2 = DepthwiseSeparableConv2d(out_channels, out_channels,
                                                kernel_size, dilation, depth_multiplier)
        self.use_projection = (in_channels != out_channels)
        if self.use_projection:
            self.projection = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = self.projection(x) if self.use_projection else x
        out = self.dsconv1(F.relu(self.bn1(x)))
        out = self.dsconv2(F.relu(self.bn2(out)))
        return out + identity


class DeepRx(nn.Module):
    DEFAULT_BLOCK_CONFIGS = [
        (64, 1, 1), (64, 1, 1), (128, 2, 3), (128, 2, 3),
        (256, 2, 3), (256, 3, 6), (256, 2, 3),
        (128, 2, 3), (128, 2, 3), (64, 1, 1), (64, 1, 1),
    ]

    def __init__(self, n_rx_antennas=2, max_bits_per_symbol=8,
                 depth_multiplier=2, block_configs=None):
        super().__init__()
        self.n_rx = n_rx_antennas
        self.B = max_bits_per_symbol
        self.depth_multiplier = depth_multiplier
        if block_configs is None:
            block_configs = self.DEFAULT_BLOCK_CONFIGS

        n_input_channels = 2 * (2 * n_rx_antennas + 1)
        first_ch = block_configs[0][0]
        self.conv_in = nn.Conv2d(
            n_input_channels, first_ch, kernel_size=3, padding=1, bias=False)

        self.blocks = nn.ModuleList()
        in_ch = first_ch
        for out_ch, dil_s, dil_f in block_configs:
            self.blocks.append(PreactivationResNetBlock(
                in_ch, out_ch, (3, 3), (dil_s, dil_f), depth_multiplier))
            in_ch = out_ch

        self.conv_out = nn.Conv2d(
            block_configs[-1][0], self.B, kernel_size=1, bias=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        out = self.conv_in(z)
        for block in self.blocks:
            out = block(out)
        return self.conv_out(out)

    @torch.no_grad()
    def detect_bits(self, z):
        return (self.forward(z) > 0).float()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_summary(self):
        n_in = 2 * (2 * self.n_rx + 1)
        print("=" * 80)
        print(f"{'DeepRx Architecture Summary':^80}")
        print("=" * 80)
        c = self.conv_in.out_channels
        print(
            f"  Input: (batch, {n_in}, S, F) -> Conv In -> (batch, {c}, S, F)")
        for i, block in enumerate(self.blocks):
            out_ch = block.dsconv1.pointwise.out_channels
            d = block.dsconv1.depthwise.dilation
            proj = ' [projection]' if block.use_projection else ''
            print(f"  ResNet Block {i+1:>2}: dilation={d}, out={out_ch}{proj}")
        print(f"  Conv Out: (batch, {self.B}, S, F)")
        print(f"  Total Parameters: {self.count_parameters():,}")
        print("=" * 80)


def build_deeprx_input(rx_signal, tx_pilots):
    Nr = rx_signal.shape[1]
    tx_pilots_expanded = tx_pilots.expand(-1, Nr, -1, -1)
    raw_ch_est = rx_signal * torch.conj(tx_pilots_expanded)
    Z_complex = torch.cat([rx_signal, tx_pilots, raw_ch_est], dim=1)
    Z = torch.cat([Z_complex.real, Z_complex.imag], dim=1)
    return Z


class DeepRxLoss(nn.Module):
    def forward(self, logits, target_bits, data_mask, bit_mask):
        # CRITICAL: expand mask to match logits shape including batch dimension
        full_mask = (data_mask * bit_mask).expand_as(logits)
        bce = F.binary_cross_entropy_with_logits(
            logits, target_bits, reduction='none')
        return (bce * full_mask).sum() / full_mask.sum().clamp(min=1.0)


MODULATION_CONFIG = {'QPSK': 2, '16QAM': 4, '64QAM': 6, '256QAM': 8}


def create_bit_mask(modulation, B=8, device='cpu'):
    n_bits = MODULATION_CONFIG[modulation]
    mask = torch.zeros(1, B, 1, 1, device=device)
    mask[0, :n_bits, 0, 0] = 1.0
    return mask


def create_pilot_mask(S=14, F=312, config='2_pilots_A', device='cpu'):
    mask = torch.zeros(1, 1, S, F, device=device)
    if config == '1_pilot_A':
        mask[0, 0, 2, 0::2] = 1.0
    elif config == '1_pilot_B':
        mask[0, 0, 2, 1::2] = 1.0
    elif config == '2_pilots_A':
        mask[0, 0, 2, 0::2] = 1.0
        mask[0, 0, 11, 1::2] = 1.0
    elif config == '2_pilots_B':
        mask[0, 0, 2, 1::2] = 1.0
        mask[0, 0, 11, 0::2] = 1.0
    else:
        raise ValueError(f"Unknown config: {config}")
    return mask


def generate_qpsk_pilots(batch, S, F, pilot_mask, device='cpu'):
    signs_r = 2 * torch.randint(0, 2, (batch, 1, S, F),
                                device=device).float() - 1
    signs_i = 2 * torch.randint(0, 2, (batch, 1, S, F),
                                device=device).float() - 1
    qpsk = (1.0 / math.sqrt(2)) * torch.complex(signs_r, signs_i)
    return qpsk * pilot_mask


def compute_ber(logits, target_bits, data_mask, bit_mask):
    # CRITICAL: expand mask to match logits shape including batch dimension
    full_mask = (data_mask * bit_mask).expand_as(logits)
    detected = (logits > 0).float()
    n_errors = ((detected != target_bits).float() * full_mask).sum()
    n_total = full_mask.sum().clamp(min=1.0)
    return (n_errors / n_total).item()


# ═══════════════════════════════════════════════════════════════════════════
#  VERIFICATION — with strict BER checks
# ═══════════════════════════════════════════════════════════════════════════

def run_verification():
    print("\n" + "=" * 70)
    print(f"{'DeepRx FINAL Verification':^70}")
    print("=" * 70)

    batch_size, Nr, S, F, B = 4, 2, 14, 312, 8
    modulation = '16QAM'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}, Batch: {batch_size}")

    # ── Test 1: Masks ──
    print(f"\n  Test 1: Masks")
    pilot_mask = create_pilot_mask(S, F, '2_pilots_A', device)
    data_mask = 1.0 - pilot_mask
    print(f"    Pilots: {int(pilot_mask.sum())}, Data: {int(data_mask.sum())}")
    assert int(pilot_mask.sum()) + int(data_mask.sum()) == S * F
    print("    PASSED")

    # ── Test 2: Input ──
    print(f"\n  Test 2: Input Construction")
    rx_signal = torch.randn(batch_size, Nr, S, F,
                            dtype=torch.cfloat, device=device)
    tx_pilots = generate_qpsk_pilots(batch_size, S, F, pilot_mask, device)
    Z = build_deeprx_input(rx_signal, tx_pilots)
    Nc = 2 * Nr + 1
    assert Z.shape == (batch_size, 2*Nc, S, F)
    print(f"    Z: {Z.shape}")
    print("    PASSED")

    # ── Test 3: Model ──
    print(f"\n  Test 3: Model")
    model = DeepRx(n_rx_antennas=Nr, max_bits_per_symbol=B).to(device)
    model.print_summary()
    print("    PASSED")

    # ── Test 4: Forward ──
    print(f"\n  Test 4: Forward Pass")
    model.eval()
    with torch.no_grad():
        logits = model(Z)
    assert logits.shape == (batch_size, B, S, F)
    print(f"    Output: {logits.shape}")
    print("    PASSED")

    # ── Test 5: CRITICAL BER Convention ──
    print(f"\n  Test 5: BER Convention (CRITICAL)")
    target_bits = torch.randint(
        0, 2, (batch_size, B, S, F), device=device, dtype=torch.float32)
    bit_mask = create_bit_mask(modulation, B, device)

    # Perfect logits: +5 for bit=1, -5 for bit=0
    perfect_logits = (target_bits * 2 - 1) * 5.0
    ber_perfect = compute_ber(perfect_logits, target_bits, data_mask, bit_mask)
    print(f"    Perfect logits BER:  {ber_perfect:.6f} (MUST be 0.0)")
    assert ber_perfect == 0.0, f"FATAL: BER should be 0, got {ber_perfect}"

    # Inverted logits
    ber_inverted = compute_ber(-perfect_logits,
                               target_bits, data_mask, bit_mask)
    print(f"    Inverted logits BER: {ber_inverted:.4f} (MUST be ~1.0)")
    assert 0.99 <= ber_inverted <= 1.0, f"FATAL: BER should be ~1.0, got {ber_inverted}"

    # Random logits
    random_logits = torch.randn_like(target_bits)
    ber_random = compute_ber(random_logits, target_bits, data_mask, bit_mask)
    print(f"    Random logits BER:   {ber_random:.4f} (MUST be ~0.5)")
    assert 0.35 <= ber_random <= 0.65, f"FATAL: BER should be ~0.5, got {ber_random}"

    print("    PASSED")

    # ── Test 6: Loss ──
    print(f"\n  Test 6: Loss")
    criterion = DeepRxLoss()
    loss = criterion(logits, target_bits, data_mask, bit_mask)
    print(f"    Loss: {loss.item():.4f} (should be 0.5-1.5)")
    assert 0.1 < loss.item() < 5.0, f"Loss out of range: {loss.item()}"
    print("    PASSED")

    # ── Test 7: Gradient ──
    print(f"\n  Test 7: Gradient Flow")
    model.train()
    model.zero_grad()
    logits = model(Z)
    loss = criterion(logits, target_bits, data_mask, bit_mask)
    loss.backward()
    n_grad = sum(1 for p in model.parameters()
                 if p.requires_grad and p.grad is not None)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_grad == n_total
    print(f"    Gradients: {n_grad}/{n_total}")
    print("    PASSED")

    # ── Test 8: Multi-Modulation with BER CHECK ──
    print(f"\n  Test 8: Multi-Modulation (BER must be 0-1)")
    model.eval()
    with torch.no_grad():
        logits = model(Z)
        for mod_name, n_bits in MODULATION_CONFIG.items():
            bm = create_bit_mask(mod_name, B, device)
            loss_val = criterion(logits, target_bits, data_mask, bm)
            ber_val = compute_ber(logits, target_bits, data_mask, bm)
            print(
                f"    {mod_name:>7}: bits={n_bits}, Loss={loss_val.item():.4f}, BER={ber_val:.4f}")

            # THIS MUST PASS - if BER > 1.0 something is fundamentally wrong
            assert 0.0 <= ber_val <= 1.0, f"FATAL: BER={ber_val} for {mod_name}! Must be 0-1"

    print("    PASSED")

    # ── Test 9: Variable Size ──
    print(f"\n  Test 9: Variable Input Size")
    for test_F in [48, 156, 312]:
        z_var = torch.randn(2, 2*Nc, S, test_F, device=device)
        with torch.no_grad():
            out = model(z_var)
        assert out.shape == (2, B, S, test_F)
        print(f"    F={test_F}: OK")
    print("    PASSED")

    print(f"\n{'='*70}")
    print(f"{'ALL TESTS PASSED - FILE IS CORRECT':^70}")
    print(f"{'='*70}\n")
    return model


if __name__ == "__main__":
    run_verification()
