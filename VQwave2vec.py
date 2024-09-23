import torch
import torchaudio
import os
import sys


# # # # Get the path to the root directory of 'vencoder'
# vencoder_root_dir = os.path.abspath('vencoder/wav2vec')
# src_dir = os.path.join(vencoder_root_dir, 'src')

# # # Add the root directory and src directory to the system path
# sys.path.insert(0, vencoder_root_dir)
# sys.path.insert(0, src_dir)

from models.wav2vec import Wav2VecModel

class VQWave2Vec:
    def __init__(self, model_path="pretrain/vqwave2vec/vq-wav2vec.pt", device=None):
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model = Wav2VecModel.build_model(checkpoint['args'], task=None)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.model = self.model.to(self.dev)

    def encoder(self, audio_data):
        """
        Get quantized codes from audio data using VQ-Wave2Vec.

        Args:
            audio_data (torch.Tensor): Audio data tensor with shape (batch_size, num_samples).

        Returns:
            torch.Tensor: Quantized codes with shape (batch_size, num_codes).
        """
        with torch.no_grad():
            
            # Convert the NumPy array to a PyTorch tensor
            #audio_data = torch.tensor(audio_data)
            # Move input tensor to the same device as the model's weight tensor
           # audio_data = audio_data.to(self.dev)
            # Reshape the tensor to (batch_size, num_samples, 1)
            feats = audio_data
            if feats.dim() == 2:  # double channels
                feats = feats.mean(-1)
            assert feats.dim() == 1, feats.dim()
            feats = feats.view(1, -1)

            z = self.model.feature_extractor(feats)
            _, idxs = self.model.vector_quantizer.forward_idx(z)
            return idxs
