import argparse
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from random import shuffle

import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

import diffusion.logger.utils as du
import utils
from diffusion.vocoder import Vocoder
from modules.mel_processing import spectrogram_torch, create_spectrogram

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

hps = utils.get_hparams_from_file("configs/config.json")
dconfig = du.load_config("configs/diffusion.yaml")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length
speech_encoder = hps["model"]["speech_encoder"]

import transform_content_features as tcf

import torch.nn as nn

import librosa
import torchaudio
import torchaudio.transforms as T


import torch.nn.functional as F

def process_one(filename, contentvec, whisper, wenet, prosody, f0p, device, diff=False, mel_extractor=None):\

    # Load audio file
    wav, sr = librosa.load(filename, sr=sampling_rate)
    audio_norm = torch.FloatTensor(wav).unsqueeze(0).to(device)

    # Resample to 16kHz
    wav16kresamp = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
    wav16k = torch.from_numpy(wav16kresamp).to(device)

    # ContentVec
    contentVecFeat = contentvec.encoder(wav16k)
    # Save the contentvc features
    soft_path = filename + ".soft.pt"
    torch.save(contentVecFeat.cpu(), soft_path)
   
    # Whisper
    whisperFeat = whisper.encoder(wav16k)
    #save the whsiper feature
    whisp_path = filename + ".whis.pt"
    torch.save(whisperFeat.cpu(), whisp_path)

    # #wenet
    wenetFeat = wenet.encoder(wav16k)
    wenet_path = filename + ".wenet.pt"
    torch.save(wenetFeat.cpu(), wenet_path)

    # get prosody feature
    prosody_path = filename + ".pros.pt"
    prosodyFeat = prosody.encoder(wav16k)
    torch.save(prosodyFeat.cpu(), prosody_path)

    # F0 prediction data
    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        f0_predictor = utils.get_f0_predictor(f0p, sampling_rate=sampling_rate, hop_length=hop_length, device=device, threshold=0.05)
        f0, uv = f0_predictor.compute_f0_uv(wav)
        np.save(f0_path, np.asanyarray((f0, uv), dtype=object))

    # Spectrogram data
    spec_path = filename.replace(".wav", ".spec.pt")
    if not os.path.exists(spec_path):
        if sr != hps.data.sampling_rate:
            raise ValueError(f"{sr} SR doesn't match target {hps.data.sampling_rate} SR")

        spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, center=False)
        spec = torch.squeeze(spec, 0)
        torch.save(spec, spec_path)

    if diff or hps.model.vol_embedding:
        volume_path = filename + ".vol.npy"
        volume_extractor = utils.Volume_Extractor(hop_length)
        if not os.path.exists(volume_path):
            volume = volume_extractor.extract(audio_norm)
            np.save(volume_path, volume.cpu().numpy())

    if diff:
        mel_path = filename + ".mel.npy"
        if not os.path.exists(mel_path) and mel_extractor is not None:
            mel_t = mel_extractor.extract(audio_norm, sampling_rate)
            mel = mel_t.squeeze().cpu().numpy()
            np.save(mel_path, mel)
        aug_mel_path = filename + ".aug_mel.npy"
        aug_vol_path = filename + ".aug_vol.npy"
        max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
        max_shift = min(1, np.log10(1 / max_amp))
        log10_vol_shift = random.uniform(-1, max_shift)
        keyshift = random.uniform(-5, 5)
        if mel_extractor is not None:
            aug_mel_t = mel_extractor.extract(audio_norm * (10 ** log10_vol_shift), sampling_rate, keyshift=keyshift)
        aug_mel = aug_mel_t.squeeze().cpu().numpy()
        aug_vol = volume_extractor.extract(audio_norm * (10 ** log10_vol_shift))
        if not os.path.exists(aug_mel_path):
            np.save(aug_mel_path, np.asanyarray((aug_mel, keyshift), dtype=object))
        if not os.path.exists(aug_vol_path):
            np.save(aug_vol_path, aug_vol.cpu().numpy())

def process_batch(file_chunk, f0p, diff=False, mel_extractor=None, device="cpu"):
    logger.info("Loading speech encoder for content...")
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"Rank {rank} uses device {device}")

    # Load speech encoders
    contentvec = utils.get_speech_encoder("vec768l12")
    whisper = utils.get_speech_encoder("whisper-ppg")
    wenet = utils.get_wenet_encoder()
    prosody = utils.get_prosody_encoder()

    logger.info(f"Loaded speech encoders for rank {rank}")
    for filename in tqdm(file_chunk, position=rank):
        try:
            
            process_one(filename, contentvec, whisper, wenet, prosody, f0p, device, diff, mel_extractor)
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}", exc_info=True)

def parallel_process(filenames, num_processes, f0p, diff, mel_extractor, device):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filenames) / num_processes)
            end = int((i + 1) * len(filenames) / num_processes)
            file_chunk = filenames[start:end]
            tasks.append(executor.submit(process_batch, file_chunk, f0p, diff, mel_extractor, device=device))
        for task in tqdm(tasks, position = 0):
            task.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default=None)
    parser.add_argument(
        "--in_dir", type=str, default="dataset/44k", help="path to input dir"
    )
    parser.add_argument(
        '--use_diff',action='store_true', help='Whether to use the diffusion model'
    )
    parser.add_argument(
        '--f0_predictor', type=str, default="rmvpe", help='Select F0 predictor, can select crepe,pm,dio,harvest,rmvpe,fcpe|default: pm(note: crepe is original F0 using mean filter)'
    )
    parser.add_argument(
        '--num_processes', type=int, default=1, help='You are advised to set the number of processes to the same as the number of CPU cores'
    )
    args = parser.parse_args()
    f0p = args.f0_predictor
    device = args.device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(speech_encoder)
    logger.info("Using device: " + str(device))
    logger.info("Using SpeechEncoder: " + speech_encoder)
    logger.info("Using extractor: " + f0p)
    logger.info("Using diff Mode: " + str(args.use_diff))

    if args.use_diff:
        print("use_diff")
        print("Loading Mel Extractor...")
        mel_extractor = Vocoder(dconfig.vocoder.type, dconfig.vocoder.ckpt, device=device)
        print("Loaded Mel Extractor.")
    else:
        mel_extractor = None
    filenames = glob(f"{args.in_dir}/*/*.wav", recursive=True)  # [:10]
    shuffle(filenames)
    mp.set_start_method("spawn", force=True)

    num_processes = args.num_processes
    if num_processes == 0:
        num_processes = os.cpu_count()

    parallel_process(filenames, num_processes, f0p, args.use_diff, mel_extractor, device)

