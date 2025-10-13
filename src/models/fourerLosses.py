import torch
import torch.fft as fft
import torch.nn as nn
from torch.nn import functional as F
import math


class FourierLossETH(nn.Module):  # nn.Module
    def __init__(self, reduction='mean', **kwargs):
        super(FourierLossETH, self).__init__()
        self.reduction = reduction

    def forward(self, sr, hr):
        sr_w, hr_w = self.addWindows(sr, hr)
        sr_amplitudes, sr_phases = self.comFourier(sr_w)
        hr_amplitudes, hr_phases = self.comFourier(hr_w)
        amp_loss = self.get_l1loss(sr_amplitudes, hr_amplitudes)
        phase_loss = self.get_angleloss(sr_phases, hr_phases)
        
        return phase_loss, amp_loss

    def comFourier(self, image):
        Fs = fft.fftshift(fft.fftn(image, dim=(2, 3)), dim=(2, 3))
        amplitudes = Fs.abs()
        phases = torch.angle(Fs)
        return amplitudes, phases

    def addWindows(self, sr, hr):
        b, c, h, w = sr.size()
        win1 = torch.hann_window(h).reshape(h, 1)
        win2 = torch.hann_window(w).reshape(1, w)
        win = torch.mm(win1, win2)
        sr, hr = sr * win, hr * win
        return sr, hr

    def get_angleloss(self, pred, target):
        """
        pred_phase, true_phase: complex phases in (-π, π] with shape [B, C, H, W]

        Computes     2/N * Σ_{u=0}^{H/2-1} Σ_v  | d(Δφ) |
        where        d(Δφ) = min(|Δφ|, 2π-|Δφ|)  ∈ [0, π].
        """
        # 1. signed difference wrapped to (-π, π]
        diff = (pred - target + math.pi) % (2*math.pi) - math.pi   # [-π, π)

        # 2. shortest distance on the circle, 0 … π
        d = torch.abs(diff)
        d = torch.minimum(d, 2*math.pi - d)

        # 3. keep only the positive-frequency half along the first spatial axis
        d = d[..., : d.shape[-2] // 2, :]

        # 4. L1 reduction and factor 2
        loss = d.mean()

        return loss

    def get_l1loss(self, pred, target):
        # absolute difference
        diff = torch.abs(pred - target)                # [B,C,H,W]

        # positive-frequency half-plane (rows 0 … H/2-1)
        diff = diff[..., : diff.shape[-2] // 2, :]           # [B,C,H/2,W]

        # L1 over that half and factor 2
        loss = diff.mean()

        return loss
    

class CRPSloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, gt):
        """
        preds: Tensor of shape [B, N, C, H, W]
        gt:    Tensor of shape [B, C, H, W]
        """
        B, N, C, H, W = preds.shape
        obs = gt.unsqueeze(1)                          # [B, 1, C, H, W]

        # Term 1: mean |f_i − x|
        term1 = torch.abs(preds - obs).mean(dim=1)     # [B, C, H, W]

        # Term 2: unbiased penalisation of ensemble spread
        if N > 1:
            # |f_i − f_j| for all i, j
            pairwise = torch.abs(preds.unsqueeze(2) - preds.unsqueeze(1))  # [B, N, N, C, H, W]

            # Remove diagonal (i = j)
            mask = ~torch.eye(N, dtype=torch.bool, device=preds.device).view(1, N, N, 1, 1, 1)
            pairwise = pairwise * mask

            term2 = pairwise.sum(dim=(1, 2))           # Σ_{i≠j} |f_i − f_j|
            term2 = term2 / (2 * N * (N - 1))          # ½ N(N−1) denominator
        else:                                          # single-member “ensemble”
            term2 = torch.zeros_like(term1)

        crps_field = term1 - term2                     # [B, C, H, W]
        return crps_field.mean()                       # scalar suitable for loss.backward()
    
    
class FourierLossHK(nn.Module):
    def __init__(self, reduction='mean', **kwargs):
        super(FourierLossHK, self).__init__()
        self.reduction = reduction

    def forward(self, sr, hr):
        sr_w, hr_w = self.addWindows(sr, hr)
        sr_F = self.comFourier(sr_w)
        hr_F = self.comFourier(hr_w)
        amp_loss = self.get_amplitude_loss(sr_F, hr_F)
        corr_loss = self.get_fourier_correlation_loss(sr_F, hr_F)
        return corr_loss, amp_loss

    def comFourier(self, image):
        return fft.fftshift(fft.fftn(image, dim=(2, 3)), dim=(2, 3))

    def addWindows(self, sr, hr):
        b, c, h, w = sr.size()
        win1 = torch.hann_window(h, device=sr.device).reshape(h, 1)
        win2 = torch.hann_window(w, device=sr.device).reshape(1, w)
        win = torch.mm(win1, win2)
        sr, hr = sr * win, hr * win
        return sr, hr

    def get_fourier_correlation_loss(self, predF, targetF):
        """
        Computes 1 - cosine similarity between complex Fourier coefficients:
        L = 1 - Re( <F, F̂> ) / ( ||F|| * ||F̂|| )
        """
        # Flatten the spatial dimensions
        predF = predF[..., : predF.shape[-2] // 2, :]  # keep positive frequencies
        targetF = targetF[..., : targetF.shape[-2] // 2, :]

        dot = (predF * targetF.conj()).real.sum(dim=(-2, -1))  # [B, C]
        norm_pred = predF.abs().pow(2).sum(dim=(-2, -1)).sqrt() 
        norm_target = targetF.abs().pow(2).sum(dim=(-2, -1)).sqrt() 
        cosine = dot / (norm_pred * norm_target)

        loss = (1.0 - cosine).mean()
        return loss

    def get_amplitude_loss(self, predF, targetF):
        pred = predF.abs()
        target = targetF.abs() 
        diff = (pred - target) ** 2
        diff = diff[..., : diff.shape[-2] // 2, :]
        loss = diff.mean()
        return loss
    
    
class FourierLossMSEAmpHF(nn.Module):
    def __init__(self, reduction='mean', **kwargs):
        super(FourierLossMSEAmpHF, self).__init__()
        self.reduction = reduction

    def forward(self, sr, hr):
        sr_w, hr_w = self.addWindows(sr, hr)
        sr_F = self.comFourier(sr_w)
        hr_F = self.comFourier(hr_w)

        # Build frequency weights dynamically from shape
        freq_weights = self.make_high_freq_weights(sr_F.shape, sr.device)

        pix_loss = self.get_pixel_mse_loss(sr, hr)
        amp_loss = self.get_log_amplitude_loss(sr_F, hr_F, freq_weights)
        return pix_loss, amp_loss

    def comFourier(self, image):
        return fft.fftshift(fft.fftn(image, dim=(2, 3)), dim=(2, 3))

    def addWindows(self, sr, hr):
        b, c, h, w = sr.size()
        win1 = torch.hann_window(h, device=sr.device).reshape(h, 1)
        win2 = torch.hann_window(w, device=sr.device).reshape(1, w)
        win = torch.mm(win1, win2)
        sr, hr = sr * win, hr * win
        return sr, hr

    def get_pixel_mse_loss(self, pred, target):
        return torch.mean((pred - target) ** 2)

    def get_log_amplitude_loss(self, predF, targetF, freq_weights):
        # pred_log = torch.log(predF.abs())
        # target_log = torch.log(targetF.abs())
        pred_phase = predF.abs()
        target_phase = targetF.abs()
        diff = (pred_phase - target_phase) ** 2

        # Positive-frequency half-plane
        diff = diff[..., : diff.shape[-2] // 2, :]
        freq_weights = freq_weights[..., : diff.shape[-2], :]

        weighted = diff * freq_weights
        return torch.mean(weighted)

    def make_high_freq_weights(self, shape, device):
        # shape: [B, C, H, W] of FFT image
        _, _, H, W = shape
        u = torch.arange(H, device=device).reshape(H, 1).float()
        v = torch.arange(W, device=device).reshape(1, W).float()
        center_u = (H - 1) / 2
        center_v = (W - 1) / 2

        r = torch.sqrt((u - center_u)**2 + (v - center_v)**2)
        r = r / r.max()  # normalize to [0, 1]

        # Emphasize high frequencies (square to exaggerate)
        w = r**2
        return w.unsqueeze(0).unsqueeze(0)  # shape [1,1,H,W]
    
    
class FourierLossCarlo(nn.Module):
    def __init__(self):
        super(FourierLossCarlo, self).__init__()
        self.dx = 5.5 # grid spacing in the physical dimension
        self.eps = 1e-8  # small value to avoid log(0)
        

    def forward(self, sr, hr):
        """
        sr : Tensor of shape [B, C, H, W] (super-resolved image)
        hr : Tensor of shape [B, C, H, W] (high-resolution ground truth)
        """
        
        sr_psd = self.getpsd(sr)
        hr_psd = self.getpsd(hr)
        diff_log_psd = torch.log(sr_psd + self.eps) - torch.log(hr_psd + self.eps)
        w = self.make_high_freq_weights(sr)
        mse_loss = torch.mean((sr - hr) ** 2)
        psd_loss = torch.sqrt(torch.mean(w * diff_log_psd ** 2))
        return mse_loss, psd_loss

    def getpsd(self, image):
        """
        Computes the 2D power spectral density (PSD) of real-valued images.
        
        Parameters:
            image : Tensor of shape [B, C, H, W]
            self.dx : spatial sampling interval (scalar)

        Returns:
            psd : Tensor of shape [B, C, H, W//2+1] (after rFFTN)
        """
        # 2D real FFT
        v_ft = torch.fft.rfftn(image, dim=(2, 3))

        # Compute power (|F|^2) and normalize by spatial size and dx
        N = image.shape[-2] * image.shape[-1]
        psd = (v_ft.real**2 + v_ft.imag**2) / (N * self.dx)

        return psd
    
    def make_high_freq_weights(self, image):
        """
        Creates a 2D weight map that emphasizes high spatial frequencies,
        aligned with rFFTN output: shape [B, C, H, W//2+1]
        """
        _, _, H, W = image.shape
        W_rfft = W // 2 + 1

        dk_h = 2 * math.pi / (H * self.dx)
        dk_w = 2 * math.pi / (W * self.dx)

        k_h = torch.arange(H, device=image.device).reshape(H, 1) * dk_h  # vertical
        k_w = torch.arange(W_rfft, device=image.device).reshape(1, W_rfft) * dk_w  # horizontal

        k_grid = torch.sqrt(k_h ** 2 + k_w ** 2)  # shape: [H, W//2+1]
        weights = (k_grid / k_grid.max()).pow(2)  # quadratic emphasis

        return weights.unsqueeze(0).unsqueeze(0)  # shape [1, 1, H, W//2+1]
    
    
    
    


if __name__ == '__main__':
    sr_tensor = torch.rand([4, 5, 368, 368])
    hr_tensor = torch.rand([4, 5, 368, 368])
    F_ETH = FourierLossETH()
    F_Delft = CRPSloss()
    F_HK = FourierLossHK()
    F_Carlo = FourierLossCarlo()
    phase_loss_ETH, amp_loss_ETH = F_ETH(sr_tensor, hr_tensor)
    phase_loss_Delft, amp_loss_Delft = F_Delft(sr_tensor, hr_tensor)
    corr_loss_HK, amp_loss_HK = F_HK(sr_tensor, hr_tensor)
    pix_loss_, amp_loss_ = F_Carlo(sr_tensor, hr_tensor)
    
    
    print(f'ETH Phase Loss: {phase_loss_ETH.item()}, Amplitude Loss: {amp_loss_ETH.item()}')
    print(f'Delft Phase Loss: {phase_loss_Delft.item()}, Amplitude Loss: {amp_loss_Delft.item()}')
    print(f'HK Correlation Loss: {corr_loss_HK.item()}, Amplitude Loss: {amp_loss_HK.item()}')
    print(f'MSE Pixel Loss: {pix_loss_.item()}, Amplitude Loss: {amp_loss_.item()}')
    a=1
    
    
