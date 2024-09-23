# prosody-vits-svc

Here presents code examples for 'prosody-vits-svc' which is an update to the original 'so-vits-svc' with the inclusion of:

1) fused multiple contenmt features from: https://github.com/open-mmlab/Amphion/blob/main/egs/svc/README.md
2) Learnt prosody features in training from the target voice extracted from VQ-Wav2vec which undergoes encoding and filtering and combined with energy
3) Mutual information (MI) loss integrated to assist in the distentanglement between prosody, content and melody inspired by: https://github.com/Wendison/VQMIVC
