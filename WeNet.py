import torch
from wenet.utils.init_model import init_model
from wenet.utils.checkpoint import load_checkpoint
import yaml
import os
import copy
from torchaudio.compliance import kaldi

"""
code adapted from: https://github.com/open-mmlab/Amphion/blob/main/processors/content_extractor.py
"""

class WeNet:
    def __init__(self, model_path="pretrain/wenet/20220506_u2pp_conformer_exp_wenetspeech/final.pt", config_path="pretrain/wenet/20220506_u2pp_conformer_exp_wenetspeech/train.yaml", device=None):

        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)

        self.model = self.load_model(model_path, config_path)
        
        self.model.to(self.dev)
        self.model.eval()

    def load_model(self, model_path, config_path):

        # load Wenet config
        with open(config_path, "r") as w:
            wenet_configs = yaml.load(w, Loader=yaml.FullLoader)

        self.extract_conf = copy.deepcopy(wenet_configs["dataset_conf"])
        print("Loading Wenet Model...")
        model, configs = init_model(args= '',configs=wenet_configs)
        load_checkpoint(model, model_path)
        
        return model

    def encoder(self, audio):
        
        audio = audio.to(self.dev)

        with torch.no_grad():
            feats = audio
            if feats.dim() == 2:  # double channels
                feats = feats.mean(-1)  # Convert to mono by averaging the channels
            assert feats.dim() == 1, feats.dim()
            feats = feats.view(1, -1)  # Reshape to (1, T)

            # Extract fbank/mfcc features using Kaldi
            assert self.extract_conf is not None, "Load model first!"
            feats_type = self.extract_conf.get("feats_type", "fbank")
            assert feats_type in ["fbank", "mfcc"]

            if feats_type == "fbank":
                fbank_conf = self.extract_conf.get("fbank_conf", {})
                feat = kaldi.fbank(
                    feats,
                    sample_frequency=16000,
                    num_mel_bins=fbank_conf["num_mel_bins"],
                    frame_length=fbank_conf["frame_length"],
                    frame_shift=fbank_conf["frame_shift"],
                    dither=fbank_conf["dither"],
                )
            elif feats_type == "mfcc":
                mfcc_conf = self.extract_conf.get("mfcc", {})
                feat = kaldi.mfcc(
                    feats,
                    sample_frequency=16000,
                    num_mel_bins=mfcc_conf["num_mel_bins"],
                    frame_length=mfcc_conf["frame_length"],
                    frame_shift=mfcc_conf["frame_shift"],
                    dither=mfcc_conf["dither"],
                    num_ceps=mfcc_conf.get("num_ceps", 40),
                    high_freq=mfcc_conf.get("high_freq", 0.0),
                    low_freq=mfcc_conf.get("low_freq", 20.0),
                )

            feats_length = torch.tensor([feat.shape[0]], dtype=torch.int32).to(self.dev)
            feats_tensor = feat.unsqueeze(0).to(self.dev)  # (1, len, 80)

            features = self.model.encoder_extractor(
                feats_tensor,
                feats_length,
                decoding_chunk_size=-1,
                num_decoding_left_chunks=-1,
                simulate_streaming=False,
            )

        return features


