import argparse
import yaml
import torch
import numpy as np
from src.module import Tacotron
from src.symbols import txt2seq
from src.utils import AudioProcessor
import os

def generate_speech(args):
    config = yaml.load(open(args.config, 'r'))
    model = load_ckpt(config, args.checkpoint_path)
    seq = np.asarray(txt2seq(args.text))
    seq = torch.from_numpy(seq).unsqueeze(0)
    # Decode
    with torch.no_grad():
        mel, spec, attn = model(seq)

    if args.generate_mel:
        # Save mel spectrogram
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(os.path.join(args.output_dir, args.output + '-mel.npy'), mel, allow_pickle=False)
        return
    elif args.usingGLA:
        spec = np.load(args.linear_file)

        # Generate wav file
        ap = AudioProcessor(**config['audio'])
        wav = ap.inv_spectrogram(spec.T)
        ap.save_wav(wav, os.path.join(args.output_dir, args.output + '.wav'))

    else:
        # Generate wav file
        ap = AudioProcessor(**config['audio'])
        wav = ap.inv_spectrogram(spec[0].numpy().T)
        ap.save_wav(wav, os.path.join(args.output_dir, args.output + '.wav'))


def load_ckpt(config, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model = Tacotron(**config['model']['tacotron'])
    model.load_state_dict(ckpt['state_dict'])
    # This yeilds the best performance, not sure why
    # model.mel_decoder.eval()
    model.encoder.eval()
    model.postnet.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthesize speech')
    parser.add_argument('--text', default='Welcome to national taiwan university speech lab.', type=str, help='Text to synthesize', required=False)
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--output', default='output', type=str, help='Output path', required=False)
    parser.add_argument('--generate_mel', action='store_true')
    parser.add_argument('--usingGLA', action='store_true')
    parser.add_argument('--checkpoint-path', type=str, help='Checkpoint path', required=True)
    parser.add_argument('--config', default='config/config.yaml', type=str, help='Path to config file', required=False)
    parser.add_argument('--linear_file', default='')
    args = parser.parse_args()
    generate_speech(args)




