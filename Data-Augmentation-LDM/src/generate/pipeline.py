import os

import argparse
import yaml
import torch
from torch import autocast
from tqdm import tqdm, trange

from audio.tools import wav_to_fbank, read_wav_file
from audio.stft import TacotronSTFT
from ldm.models.diffusion.ddim import DDIMSampler
# from einops import repeat
import os
from annotator.style.model import style_net
from annotator.style_pretrained.model import style_encoder
import numpy as np

import contextlib
import wave
from einops import repeat
import scipy.io as sio

def make_batch_for_generate_audio(text, content_waveform=None, style_file_path=None, fbank=None, duration=10, batchsize=1):
    text = [text] * batchsize
    if batchsize < 1:
        print("Warning: Batchsize must be at least 1. Batchsize is set to .")
    
    if(fbank is None):
        fbank = torch.zeros((batchsize, 1024, 64))  # Not used, here to keep the code format
    else:
        fbank = torch.FloatTensor(fbank)
        fbank = fbank.expand(batchsize, 1024, 64)
        assert fbank.size(0) == batchsize
        
    stft = torch.zeros((batchsize, 1024, 512))  # Not used

    if(content_waveform is None):
        content_waveform = torch.zeros((batchsize, 160000))  # Not used
    else:
        content_waveform = torch.FloatTensor(content_waveform)
        content_waveform = content_waveform.expand(batchsize, -1)
        assert content_waveform.size(0) == batchsize

    global_conditions = []
    global_condition_weights = []

    style = 'random'

    if(style_file_path is None):
        if style == 'random':
            style_condition = torch.zeros((batchsize, 512))
        elif style == 'pretrained':
            style_condition = torch.zeros((batchsize, 1024))
    else:
        if style == 'random':
            fn_STFT = TacotronSTFT(
                1024,
                160,
                1024,
                64,
                16000,
                0,
                8000,
            )
            mel, _, _ = wav_to_fbank(
                style_file_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT
            )
            mel = mel.unsqueeze(0)
            audio_style_extractor = style_net()
            audio_style_extractor.load_state_dict(torch.load('annotator/style/style.ckpt'))
            style_condition = audio_style_extractor(mel)
        elif style == 'pretrained':
            audio_style_extractor = style_encoder()
            with torch.no_grad():
                style_condition = audio_style_extractor(style_file_path)
        style_condition = style_condition.expand(batchsize, -1)
        assert style_condition.size(0) == batchsize

    if(style_condition is not None):
        global_conditions.append(style_condition.detach())

    if len(global_conditions) != 0:
        global_conditions = np.concatenate(global_conditions)
        global_conditions = torch.from_numpy(global_conditions)
        # global_conditions = global_conditions.reshape(1,512)
        for condition in global_conditions:
            global_condition_weights.append(np.ones(1))
        global_condition_weights = np.concatenate(global_condition_weights)
        global_condition_weights = torch.from_numpy(global_condition_weights)
    
    fname = [""] * batchsize  # Not used
    fbank = fbank.unsqueeze(1)
    
    batch = dict(fbank=fbank, stft=stft, fname=fname, waveform=content_waveform, txt=text, local_conditions=[], global_conditions=global_conditions, global_condition_weights=global_condition_weights)
#     batch = (
#         fbank,
#         stft,
#         None,
#         fname,
#         waveform,
#         text,
#     )
    return batch

def get_bit_depth(fname):
    if fname[-4:] == ".mat":
        return 16
    with contextlib.closing(wave.open(fname, 'r')) as f:
        bit_depth = f.getsampwidth() * 8
        return bit_depth

def get_duration(fname):
    if fname[-4:] == ".mat":
        mat_contents = sio.loadmat(fname)
        waveform = mat_contents['struct_48kHz'][0][0][2][0]
        return waveform.shape[0] / float(48000)
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def round_up_duration(duration):
    return int(round(duration/2.5) + 1) * 2.5

def duration_to_latent_t_size(duration):
    return int(duration * 25.6)

def set_cond_audio(latent_diffusion):
    latent_diffusion.cond_stage_key = "waveform"
    latent_diffusion.cond_stage_model.embed_mode="audio"
    return latent_diffusion

def set_cond_text(latent_diffusion):
    latent_diffusion.cond_stage_key = "text"
    latent_diffusion.cond_stage_model.embed_mode="text"
    return latent_diffusion

def seed_everything(seed):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def generate_audio(
    latent_diffusion,
    text,
    content_file_path = None,
    style_file_path = None,
    seed=42,
    ddim_steps=200,
    duration=10,
    batchsize=1,
    guidance_scale=2.5,
    n_candidate_gen_per_text=3,
    config=None,
):
    seed_everything(int(seed))
    content_waveform = None
    if(content_file_path is not None):
        content_waveform = read_wav_file(content_file_path, int(duration * 102.4) * 160)
        
    batch = make_batch_for_generate_audio(text, content_waveform=content_waveform, style_file_path=style_file_path, duration=duration, batchsize=batchsize)

    latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)
    
#     if(waveform is not None):
#         print("Generate audio that has similar content as %s" % original_audio_file_path)
#         latent_diffusion = set_cond_audio(latent_diffusion)
#     else:
#         print("Generate audio using text %s" % text)
#         latent_diffusion = set_cond_text(latent_diffusion)

    with torch.no_grad():
        waveform = latent_diffusion.generate_sample(
            [batch],
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            n_candidate_gen_per_text=n_candidate_gen_per_text,
            duration=duration,
        )
    return waveform

def style_transfer(
    latent_diffusion,
    text,
    original_audio_file_path,
    transfer_strength,
    content_file_path = None,
    style_file_path = None,
    seed=42,
    duration=10,
    batchsize=1,
    guidance_scale=2.5,
    ddim_steps=200,
    config=None,
):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    assert original_audio_file_path is not None, "You need to provide the original audio file path"
    
    audio_file_duration = get_duration(original_audio_file_path)
    
    assert get_bit_depth(original_audio_file_path) == 16, "The bit depth of the original audio file %s must be 16" % original_audio_file_path
    
    # if(duration > 20):
    #     print("Warning: The duration of the audio file %s must be less than 20 seconds. Longer duration will result in Nan in model output (we are still debugging that); Automatically set duration to 20 seconds")
    #     duration = 20
    
    if(duration > audio_file_duration):
        print("Warning: Duration you specified %s-seconds must equal or smaller than the audio file duration %ss" % (duration, audio_file_duration))
        duration = round_up_duration(audio_file_duration)
        print("Set new duration as %s-seconds" % duration)

    # duration = round_up_duration(duration)
    
    # latent_diffusion = set_cond_text(latent_diffusion)

    # if config is not None:
    #     assert type(config) is str
    #     config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    # else:
    #     config = default_audioldm_config()

    seed_everything(int(seed))
    # latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)
    # latent_diffusion.cond_stage_model.embed_mode = "text"

    fn_STFT = TacotronSTFT(
        1024,
        160,
        1024,
        64,
        16000,
        0,
        8000,
    )

    mel, _, _ = wav_to_fbank(
        original_audio_file_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT
    )
    mel = mel.unsqueeze(0).unsqueeze(0).to(device)
    mel = repeat(mel, "1 ... -> b ...", b=batchsize)
    init_latent = latent_diffusion.get_first_stage_encoding(
        latent_diffusion.encode_first_stage(mel)
    )  # move to latent space, encode and sample
    if(torch.max(torch.abs(init_latent)) > 1e2):
        init_latent = torch.clip(init_latent, min=-10, max=10)
    sampler = DDIMSampler(latent_diffusion)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=1.0, verbose=False)

    t_enc = int(transfer_strength * ddim_steps)
    prompts = text

    with torch.no_grad():
        with autocast("cuda"):
            with latent_diffusion.ema_scope():
                uc = None
                if guidance_scale != 1.0:
                    uc = latent_diffusion.cond_stage_model.get_unconditional_condition(
                        batchsize
                    )

                c = latent_diffusion.get_learned_text_conditioning([prompts] * batchsize)

                # Added by Long 09/11/2023
                c = dict(c_crossattn=[uc], text_cond=[c], local_control=[], global_control=[])

                z_enc = sampler.stochastic_encode(
                    init_latent, torch.tensor([t_enc] * batchsize).to(device)
                )
                samples = sampler.decode(
                    z_enc,
                    c,
                    t_enc,
                    unconditional_guidance_scale=guidance_scale,
                    unconditional_conditioning=uc,
                )
                # x_samples = latent_diffusion.decode_first_stage(samples) # Will result in Nan in output
                # print(torch.sum(torch.isnan(samples)))
                x_samples = latent_diffusion.decode_first_stage(samples)
                # print(x_samples)
                x_samples = latent_diffusion.decode_first_stage(samples[:,:,:-3,:])
                # print(x_samples)
                waveform = latent_diffusion.first_stage_model.decode_to_waveform(
                    x_samples
                )

    return waveform

# def super_resolution_and_inpainting(
#     latent_diffusion,
#     text,
#     original_audio_file_path = None,
#     seed=42,
#     ddim_steps=200,
#     duration=None,
#     batchsize=1,
#     guidance_scale=2.5,
#     n_candidate_gen_per_text=3,
#     time_mask_ratio_start_and_end=(0.10, 0.15), # regenerate the 10% to 15% of the time steps in the spectrogram
#     # time_mask_ratio_start_and_end=(1.0, 1.0), # no inpainting
#     # freq_mask_ratio_start_and_end=(0.75, 1.0), # regenerate the higher 75% to 100% mel bins
#     freq_mask_ratio_start_and_end=(1.0, 1.0), # no super-resolution
#     config=None,
# ):
#     seed_everything(int(seed))
#     if config is not None:
#         assert type(config) is str
#         config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
#     else:
#         config = default_audioldm_config()
#     fn_STFT = TacotronSTFT(
#         config["preprocessing"]["stft"]["filter_length"],
#         config["preprocessing"]["stft"]["hop_length"],
#         config["preprocessing"]["stft"]["win_length"],
#         config["preprocessing"]["mel"]["n_mel_channels"],
#         config["preprocessing"]["audio"]["sampling_rate"],
#         config["preprocessing"]["mel"]["mel_fmin"],
#         config["preprocessing"]["mel"]["mel_fmax"],
#     )
    
#     # waveform = read_wav_file(original_audio_file_path, None)
#     mel, _, _ = wav_to_fbank(
#         original_audio_file_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT
#     )
    
#     batch = make_batch_for_text_to_audio(text, fbank=mel[None,...], batchsize=batchsize)
        
#     # latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)
#     latent_diffusion = set_cond_text(latent_diffusion)
        
#     with torch.no_grad():
#         waveform = latent_diffusion.generate_sample_masked(
#             [batch],
#             unconditional_guidance_scale=guidance_scale,
#             ddim_steps=ddim_steps,
#             n_candidate_gen_per_text=n_candidate_gen_per_text,
#             duration=duration,
#             time_mask_ratio_start_and_end=time_mask_ratio_start_and_end,
#             freq_mask_ratio_start_and_end=freq_mask_ratio_start_and_end
#         )
#     return waveform
