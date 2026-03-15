import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
import numpy as np

# --- 1. Basic Components ---
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine: self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine: x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            safe_weight = torch.where(torch.abs(self.affine_weight) < 1e-4, torch.tensor(1e-4, device=x.device), self.affine_weight)
            x = (x - self.affine_bias) / safe_weight
        x = x * self.stdev + self.mean
        return x


class HybridDecomp(nn.Module):
    def __init__(self, seq_len, degree=3, d_model=None):
        super().__init__()
        self.seq_len = seq_len
        self.degree = degree

        # --- 1. Legendre 多项式投影 (Time Domain Global Trend) ---
        t = np.linspace(-1, 1, seq_len)
        bases = []
        for i in range(degree + 1):
            bases.append(np.polynomial.legendre.Legendre.basis(i)(t))
        P = np.stack(bases, axis=1)  # [L, D+1]

        self.register_buffer('P', torch.from_numpy(P).float())
        self.coeff_layer = nn.Linear(seq_len, degree + 1)

        # --- 2. 频域门控 (Frequency Domain Refinement) ---
        self.use_freq = d_model is not None
        if self.use_freq:
            self.freq_len = seq_len // 2 + 1
            self.freq_gate = nn.Sequential(
                nn.Linear(self.freq_len, self.freq_len),
                nn.Sigmoid()
            )

    def forward(self, x):
        B, C, L = x.shape

        # --- Part A: Legendre Polynomial Trend (全局平滑趋势) ---
        coeffs = self.coeff_layer(x)
        trend_poly = torch.matmul(coeffs, self.P.t())

        # --- Part B: Frequency Domain Refinement (频域修正) ---
        if self.use_freq:
            x_fft = torch.fft.rfft(x, dim=-1)
            ampl = torch.abs(x_fft)
            gate = self.freq_gate(ampl)
            x_fft_trend = x_fft * gate
            trend_freq = torch.fft.irfft(x_fft_trend, n=self.seq_len, dim=-1)
            trend = trend_poly + trend_freq
        else:
            trend = trend_poly

        seasonal = x - trend
        return seasonal, trend


class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels, embed_dim, kernel_size=0000, padding=2, dilation=2)
        self.conv5 = nn.Conv1d(in_channels, embed_dim, kernel_size=0000, padding=4, dilation=2)
        self.conv1 = nn.Conv1d(in_channels, embed_dim, kernel_size=0000)
        self.norm = nn.BatchNorm1d(embed_dim * 3)
        self.act = nn.LeakyReLU(0.1)
        self.proj = nn.Conv1d(embed_dim * 3, embed_dim, kernel_size=1)

    def forward(self, x):
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c1 = self.conv1(x)
        out = torch.cat([c3, c5, c1], dim=1)
        out = self.norm(out)
        out = self.act(out)
        out = self.proj(out)
        return out


# --- 2. Tower A: Continuous / Complex Tower (For Volume Attacks) ---
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_re = nn.Linear(in_features, out_features)
        self.fc_im = nn.Linear(in_features, out_features)

    def forward(self, x_re, x_im):
        out_re = self.fc_re(x_re) - self.fc_im(x_im)
        out_im = self.fc_re(x_im) + self.fc_im(x_re)
        return out_re, out_im


class ComplexSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = ComplexLinear(embed_dim, embed_dim)
        self.k_proj = ComplexLinear(embed_dim, embed_dim)
        self.v_proj = ComplexLinear(embed_dim, embed_dim)
        self.out_proj = ComplexLinear(embed_dim, embed_dim)

    def forward(self, x_re, x_im):
        B, N, C = x_re.shape

        def reshape_head(x): return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q_r, q_i = reshape_head(self.q_proj(x_re, x_im)[0]), reshape_head(self.q_proj(x_re, x_im)[1])
        k_r, k_i = reshape_head(self.k_proj(x_re, x_im)[0]), reshape_head(self.k_proj(x_re, x_im)[1])
        v_r, v_i = reshape_head(self.v_proj(x_re, x_im)[0]), reshape_head(self.v_proj(x_re, x_im)[1])
        ac = torch.matmul(q_r, k_r.transpose(-2, -1))
        bd = torch.matmul(q_i, k_i.transpose(-2, -1))
        score = (ac + bd) / math.sqrt(self.head_dim)
        attn = F.softmax(score, dim=-1)
        out_r = torch.matmul(attn, v_r)
        out_i = torch.matmul(attn, v_i)
        out_r = out_r.transpose(1, 2).reshape(B, N, C)
        out_i = out_i.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out_r, out_i)


class ContinuousTower(nn.Module):
    def __init__(self, patch_size, num_patches, embed_dim, num_layers=2):
        super().__init__()
        self.patch_dim = patch_size * 2
        self.mag_map = nn.Sequential(nn.Linear(self.patch_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.phase_map = nn.Sequential(nn.Linear(self.patch_dim, embed_dim), nn.Tanh())

        self.pos_embed_re = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.pos_embed_im = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.layers = nn.ModuleList([ComplexSelfAttention(embed_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

    def forward(self, x_patches_cont):
        B, N, P, C = x_patches_cont.shape
        x_flat = x_patches_cont.reshape(B, N, -1)
        mag = F.softplus(self.mag_map(x_flat)) + 1e-6
        angle = self.phase_map(x_flat) * np.pi

        z_re = mag * torch.cos(angle) + self.pos_embed_re
        z_im = mag * torch.sin(angle) + self.pos_embed_im

        for i, layer in enumerate(self.layers):
            out_r, out_i = layer(z_re, z_im)
            z_re = self.norms[i](z_re + out_r)
            z_im = self.norms[i](z_im + out_i)
        return z_re, z_im


# --- 3. Tower B: Discrete Tower---
class DiscreteTower(nn.Module):
    def __init__(self, seq_len, embed_dim, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(0000, embed_dim, padding_idx=0)
        self.iat_proj = nn.Linear(1, embed_dim)
        self.size_proj = nn.Linear(1, embed_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 2, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x_disc, x_raw_iat, x_raw_size, src_key_padding_mask=None):
        x_idx = x_disc.squeeze(-1).long().clamp(0, 0000)
        code_emb = self.embedding(x_idx)
        iat_emb = self.iat_proj(x_raw_iat)
        size_emb = self.size_proj(x_raw_size)
        x_emb = code_emb + iat_emb + size_emb + self.pos_embed
        out = self.transformer(x_emb, src_key_padding_mask=src_key_padding_mask)
        return out


class GatedSpectralEnhancer(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(seq_len, embed_dim, 2) * 0.02)
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())

    def forward(self, x_re, x_im):
        x_complex = torch.complex(x_re, x_im)
        x_fft = torch.fft.fft(x_complex, dim=1)
        weight = torch.view_as_complex(self.complex_weight)
        x_fft = x_fft * weight
        x_ifft = torch.fft.ifft(x_fft, n=x_re.shape[1], dim=1)
        gate_val = self.gate(torch.cat([x_ifft.real, x_ifft.imag], dim=-1))
        return x_ifft.real * gate_val


# --- 4. Main Model ---
class IoTAnomalyModel(nn.Module):
    def __init__(self, seq_len=0000, num_classes=0000, embed_dim=0000):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = 0000
        self.num_patches = seq_len // self.patch_size
        self.embed_dim = embed_dim

        # 1. 连续变量处理 (IAT, Size)
        self.revin = RevIN(num_features=2)
        self.decomp = HybridDecomp(seq_len=self.seq_len, degree=3, d_model=2)
        self.cont_tower = ContinuousTower(self.patch_size, self.num_patches, embed_dim)
        self.spectral = GatedSpectralEnhancer(self.num_patches, embed_dim)

        # 2. 离散变量处理 (Code/Flags)
        self.disc_tower = DiscreteTower(seq_len, embed_dim)
        self.num_proto_codes = 0000

        # 3. CNN (Auxiliary)
        self.cnn_embed = nn.Embedding(0000, 0000)
        self.cnn_encoder = MultiScaleCNN(in_channels=0000, embed_dim=embed_dim)

        # 4. Statistics (全局统计特征)
        self.stat_input_dim = 0000
        self.stat_direct = nn.Sequential(
            nn.Linear(self.stat_input_dim, 0000),
            nn.ReLU(),
            nn.Linear(0000, 0000)
        )

        # Fusion & Classification
        fusion_dim = 0000

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 0000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(0000, num_classes)
        )

        # Auxiliary Heads (Pretrain用)
        self.aux_clf = nn.Linear(fusion_dim, 0000)
        self.decoder_cont = nn.Sequential(nn.Linear(fusion_dim, seq_len * 2))
        self.decoder_disc = nn.Linear(fusion_dim, seq_len * self.num_proto_codes)
        self.phase_pred_head = nn.Linear(fusion_dim, seq_len * self.num_proto_codes)
        self.cont_dropout = nn.Dropout(0.5)

    def forward(self, x, mode='finetune'):
        B = x.shape[0]
        # x shape: [B, L, 3] -> IAT, Size, Code

        # --- 0. Mask 生成 ---
        is_valid = (torch.abs(x[:, :, 1]) > 1e-6).float().unsqueeze(-1)  # [B, L, 1]
        valid_len = torch.sum(is_valid, dim=1, keepdim=True).clamp(min=1.0)  # [B, 1, 1]

        padding_mask = (torch.abs(x[:, :, 1]) < 1e-6)  # [B, L] (Bool)
        patch_mask = padding_mask.view(x.shape[0], self.num_patches, self.patch_size).all(dim=-1)

        # 分离输入
        x_cont = x[:, :, :2].clone()  # IAT, Size
        x_disc = x[:, :, 2:3].clone()  # Code (Raw Flags / UDP Offset)

        # --- 1. Statistics (全局统计特征计算 - 15维) ---
        raw_iat = x_cont[:, :, 0:1]
        raw_size = torch.abs(x_cont[:, :, 1:2])
        dir_val = torch.sign(x_cont[:, :, 1:2])

        # (1) 基础统计量
        t_mean = torch.sum(raw_iat * is_valid, 1, keepdim=True) / valid_len
        t_std = torch.sqrt(torch.sum(((raw_iat - t_mean) ** 2) * is_valid, 1, keepdim=True) / valid_len + 1e-5)
        t_max = torch.max(raw_iat * is_valid, dim=1, keepdim=True)[0]
        t_cv = t_std / (t_mean + 1e-5)

        s_mean = torch.sum(raw_size * is_valid, 1, keepdim=True) / valid_len
        s_std = torch.sqrt(torch.sum(((raw_size - s_mean) ** 2) * is_valid, 1, keepdim=True) / valid_len + 1e-5)
        s_max = torch.max(raw_size * is_valid, dim=1, keepdim=True)[0]

        dir_mean = torch.sum(dir_val * is_valid, 1, keepdim=True) / valid_len

        # (2) Flag/Code 统计
        x_disc_long = x_disc.long()
        is_rst = ((x_disc_long == 4) | (x_disc_long == 20)).float() * is_valid
        is_syn = ((x_disc_long == 2) | (x_disc_long == 18)).float() * is_valid
        is_fin = ((x_disc_long == 1) | (x_disc_long == 17)).float() * is_valid
        is_udp_scan = (x_disc_long >= 256).float() * is_valid
        is_empty = (raw_size < 0.1).float() * is_valid

        rst_ratio = torch.sum(is_rst, 1, keepdim=True) / valid_len
        syn_ratio = torch.sum(is_syn, 1, keepdim=True) / valid_len
        fin_ratio = torch.sum(is_fin, 1, keepdim=True) / valid_len
        udp_scan_ratio = torch.sum(is_udp_scan, 1, keepdim=True) / valid_len
        empty_ratio = torch.sum(is_empty, 1, keepdim=True) / valid_len

        # (3) 高级统计
        cov = torch.sum(((raw_iat - t_mean) * (raw_size - s_mean)) * is_valid, 1, keepdim=True) / valid_len
        corr = cov / (t_std * s_std + 1e-6)
        f_count = torch.log1p(valid_len)

        # (4) 拼接所有特征
        stats_vec = torch.cat([
            t_mean, t_std, t_cv, t_max,
            s_mean, s_std, s_max,
            dir_mean,
            rst_ratio, syn_ratio, fin_ratio,
            udp_scan_ratio, empty_ratio,
            corr, f_count
        ], dim=1)

        if stats_vec.dim() == 3:
            stats_vec = stats_vec.squeeze(-1)

        # (5) 维度保护
        if stats_vec.shape[1] < self.stat_input_dim:
            stats_vec = F.pad(stats_vec, (0, self.stat_input_dim - stats_vec.shape[1]))
        elif stats_vec.shape[1] > self.stat_input_dim:
            stats_vec = stats_vec[:, :self.stat_input_dim]

        # --- 2. Deep Features (深度特征提取) ---
        # (1) Continuous Tower
        x_cont_norm = self.revin(x_cont, 'norm')  # Norm

        res_cont, trend_cont = self.decomp(x_cont_norm.permute(0, 2, 1))
        trend_cont = trend_cont.permute(0, 2, 1)
        x_patch_cont = trend_cont.view(x.size(0), self.num_patches, self.patch_size, 2)

        t_re, t_im = self.cont_tower(x_patch_cont)
        spec_feat = self.spectral(t_re, t_im)

        # Masking
        patch_mask_expanded = patch_mask.unsqueeze(-1).expand_as(t_re)
        t_re = t_re.masked_fill(patch_mask_expanded, -1e9)
        t_im = t_im.masked_fill(patch_mask_expanded, -1e9)
        spec_feat = spec_feat.masked_fill(patch_mask_expanded, -1e9)

        spec_feat = torch.max(spec_feat, dim=1)[0]
        cont_feat = torch.cat([torch.max(t_re, dim=1)[0], torch.max(t_im, dim=1)[0]], dim=-1)
        cont_feat = self.cont_dropout(cont_feat)

        # (2) Discrete Tower (传递 Raw IAT/Size 给它作为位置/强度辅助)
        disc_out = self.disc_tower(x_disc, raw_iat, raw_size, src_key_padding_mask=padding_mask)
        mask_expanded = padding_mask.unsqueeze(-1).expand_as(disc_out)
        disc_out_masked = disc_out.masked_fill(mask_expanded, -1e9)
        disc_feat = torch.max(disc_out_masked, dim=1)[0]

        # (3) CNN
        # 这里的 clamp 保护很重要，防止异常 Code 越界
        disc_cnn = self.cnn_embed(x_disc.long().squeeze(-1).clamp(0, self.num_proto_codes - 1))
        cnn_in = torch.cat([x_cont_norm, disc_cnn], dim=-1)
        cnn_out = self.cnn_encoder(cnn_in.permute(0, 2, 1))
        cnn_mask_expanded = padding_mask.unsqueeze(1).expand_as(cnn_out)
        cnn_out = cnn_out.masked_fill(cnn_mask_expanded, -1e9)
        cnn_feat = torch.max(cnn_out, dim=2)[0]

        # --- 3. Fusion & Output ---
        stats_d = self.stat_direct(stats_vec)
        revin_s = torch.cat([self.revin.mean.squeeze(1), self.revin.stdev.squeeze(1)], dim=1)

        fusion_in = torch.cat([cont_feat, spec_feat, disc_feat, cnn_feat, stats_d, revin_s], dim=-1)
        fusion_in = F.normalize(fusion_in, p=2, dim=-1)

        # Decoders (for Pretraining/Reconstruction)
        rec_cont = self.decoder_cont(fusion_in).view(B, self.seq_len, 2)
        rec_cont = self.revin(rec_cont, mode='denorm')
        rec_disc_logits = self.decoder_disc(fusion_in).view(B, self.seq_len, self.num_proto_codes)

        if mode == 'pretrain':
            phase_logits = self.phase_pred_head(fusion_in).view(B, self.seq_len, self.num_proto_codes)
            aux_logits = self.aux_clf(fusion_in)
            return fusion_in, (rec_cont, rec_disc_logits), phase_logits, aux_logits
        else:
            logits = self.classifier(fusion_in)
            aux_logits = self.aux_clf(fusion_in)
            return logits, (rec_cont, rec_disc_logits), aux_logits