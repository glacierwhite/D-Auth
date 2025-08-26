import torch
import resampy
import torch.nn as nn
import torchaudio
import sys
sys.path.append("/fs/scratch/pi_cr_soundsee/archive/workspace/hul2pi/Uni-ControlNet-Cross-Attention")
from deepafx_st.system import System
from deepafx_st.utils import DSPMode

class style_encoder(nn.Module):
    """Here create the network you want to use by adding/removing layers in nn.Sequential"""
    def __init__(self):
        super(style_encoder, self).__init__()
        ckpt = '/fs/scratch/pi_cr_soundsee/archive/workspace/hul2pi/Uni-ControlNet-Cross-Attention/checkpoints/style/libritts/autodiff/lightning_logs/version_1/checkpoints/epoch=367-step=1226911-val-libritts-autodiff.ckpt'
        use_dsp = False
        self.system = System.load_from_checkpoint(ckpt, dsp_mode=DSPMode.NONE, batch_size=1).eval()
        
    @torch.no_grad()
    def forward(self,filename):
        x, x_sr = torchaudio.load(filename)
        # r, r_sr = torchaudio.load(filename)

        # resample if needed
        if x_sr != 24000:
            # print("Resampling to 24000 Hz...")
            x_24000 = torch.tensor(resampy.resample(x.view(-1).numpy(), x_sr, 24000))
            x_24000 = x_24000.view(1, -1)
        else:
            x_24000 = x

        # if r_sr != 24000:
        #     print("Resampling to 24000 Hz...")
        #     r_24000 = torch.tensor(resampy.resample(r.view(-1).numpy(), r_sr, 24000))
        #     r_24000 = r_24000.view(1, -1)
        # else:
        #     r_24000 = r

        # peak normalize to -12 dBFS
        x_24000 = x_24000[0:1, : 24000 * 5]
        if x_24000.abs().max() != 0:
            x_24000 /= x_24000.abs().max()
        x_24000 *= 10 ** (-12 / 20.0)
        x_24000 = x_24000.view(1, 1, -1)

        assert torch.isnan(x_24000.view(-1)).sum().item()==0

        # # peak normalize to -12 dBFS
        # r_24000 = r_24000[0:1, : 24000 * 5]
        # r_24000 /= r_24000.abs().max()
        # r_24000 *= 10 ** (-12 / 20.0)
        # r_24000 = r_24000.view(1, 1, -1)

        # print(x_24000.shape)

        y_hat, p, e = self.system(x_24000, x_24000)

        return e