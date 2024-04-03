import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio
from torchaudio.functional import resample

## Soft Unit Encoder를 학습시키기 위해 먼저 discrete Units을 뽑는데 사용함


def encode_dataset(args): # .wav/.flac(args에서 파일형식 정의) 오디오파일로 discrete/soft units을 만들어서 .npy(numpy)파일로 저장해둔다.
    print(f"Loading hubert checkpoint")
    # load HuBERT-Soft or HuBERT-Discrete
    hubert = torch.hub.load(
        "bshall/hubert:main",
        f"hubert_{args.model}",
        trust_repo=True,
    ).cuda()

    print(f"Encoding dataset at {args.in_dir}")
    for in_path in tqdm(list(args.in_dir.rglob(f"*{args.extension}"))):
        wav, sr = torchaudio.load(in_path)
        wav = resample(wav, sr, 16000)
        wav = wav.unsqueeze(0).cuda()

        with torch.inference_mode():
            units = hubert.units(wav) #discrte units or soft units

        out_path = args.out_dir / in_path.relative_to(args.in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.with_suffix(".npy"), units.squeeze().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "model",
        help="available models (HuBERT-Soft or HuBERT-Discrete)",
        choices=["soft", "discrete"],
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio files (defaults to .flac).",
        default=".flac",
        type=str,
    )
    args = parser.parse_args()
    encode_dataset(args)
