import sys
if './' not in sys.path:
	sys.path.append('./')
	
from omegaconf import OmegaConf
import argparse

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler

from pipeline import generate_audio, style_transfer
from src.train.util import save_wave, get_time

import os

parser = argparse.ArgumentParser(description='Multi-conditioned Audio Generation')
parser.add_argument('--config-path', type=str, default='./configs/config.yaml')
# parser.add_argument('---ckpt-path', type=str, default='./training_log/lightning_logs/version_130809/checkpoints/epoch=5933-step=89000.ckpt')
# parser.add_argument('---ckpt-path', type=str, default='./training_log/lightning_logs/version_131714/checkpoints/epoch=138-step=37500.ckpt')
# parser.add_argument('---ckpt-path', type=str, default='./training_log/lightning_logs/version_131733/checkpoints/epoch=173-step=27000.ckpt')
parser.add_argument('---ckpt-path', type=str, default='./training_log/lightning_logs/version_131766/checkpoints/epoch=141-step=16000.ckpt')
# parser.add_argument('---ckpt-path', type=str, default='./ckpt/init.ckpt')
parser.add_argument('---save_path', type=str, default='./output/')

parser.add_argument(
    "--mode",
    type=str,
    required=False,
    default="generation",
    help="generation: text-to-audio generation; transfer: style transfer",
    choices=["generation", "transfer"]
)

parser.add_argument(
    "-f",
    "--file_path",
    type=str,
    required=False,
    default=None,
    help="(--mode transfer): Original audio file for style transfer; Or (--mode generation): the guidance audio file for generating simialr audio",
)

parser.add_argument(
    "--transfer_strength",
    type=float,
    required=False,
    default=0.5,
    help="A value between 0 and 1. 0 means original audio without transfer, 1 means completely transfer to the audio indicated by text",
)

parser.add_argument(
    "-t",
    "--text",
    type=str,
    required=False,
    default="",
    help="Text prompt",
)

parser.add_argument(
    "-cf",
    "--content_file_path",
    type=str,
    required=False,
    default=None,
    help="Reference audio file",
)

parser.add_argument(
    "-sf",
    "--style_file_path",
    type=str,
    required=False,
    default=None,
    help="Style audio file",
)

parser.add_argument(
    "-b",
    "--batchsize",
    type=int,
    required=False,
    default=1,
    help="Generate how many samples at the same time",
)

parser.add_argument(
    "--ddim_steps",
    type=int,
    required=False,
    default=200,
    help="The sampling step for DDIM",
)

parser.add_argument(
    "-gs",
    "--guidance_scale",
    type=float,
    required=False,
    default=2.5,
    help="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
)

parser.add_argument(
    "-dur",
    "--duration",
    type=float,
    required=False,
    default=10.0,
    help="The duration of the samples",
)

parser.add_argument(
    "-n",
    "--n_candidate_gen_per_text",
    type=int,
    required=False,
    default=3,
    help="Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier computation",
)

parser.add_argument(
    "--seed",
    type=int,
    required=False,
    default=42,
    help="Change this value (any integer number) will lead to a different generation result.",
)

args = parser.parse_args()


def main():

    config_path = args.config_path
    ckpt_path = args.ckpt_path
    save_path = args.save_path
    
    text = args.text
    random_seed = args.seed
    duration = args.duration
    guidance_scale = args.guidance_scale
    n_candidate_gen_per_text = args.n_candidate_gen_per_text
    
    os.makedirs(save_path, exist_ok=True)

    model = create_model(config_path).cpu()
    model.load_state_dict(load_state_dict(ckpt_path, location='cuda'))
    model = model.cuda()
#     ddim_sampler = DDIMSampler(model)
    
    if(args.mode == "generation"):
        waveform = generate_audio(
            model,
            text,
            args.content_file_path,
            args.style_file_path,
            random_seed,
            duration=duration,
            guidance_scale=guidance_scale,
            ddim_steps=args.ddim_steps,
            n_candidate_gen_per_text=n_candidate_gen_per_text,
            batchsize=args.batchsize,
        )

    elif(args.mode == "transfer"):
        assert args.file_path is not None
        assert os.path.exists(args.file_path), "The original audio file \'%s\' for style transfer does not exist." % args.file_path
        waveform = style_transfer(
            model,
            text,
            args.file_path,
            args.transfer_strength,
            args.content_file_path,
            args.style_file_path,
            random_seed,
            duration=duration,
            guidance_scale=guidance_scale,
            ddim_steps=args.ddim_steps,
            batchsize=args.batchsize,
        )
        waveform = waveform[:,None,:]
    
    save_wave(waveform, save_path, name="%s_%s" % (get_time(), text))


if __name__ == '__main__':
    main()