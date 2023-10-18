# BiWaveGAN2023

This repository houses code and data relating to the publication *Bidirectional Generative Adversarial Representation Learning for Natural Stimulus Synthesis.*

 - `model.py` contains definitions the model definition, in particular the `BiWaveGAN` class.
 - `train.py`  contains the training script.
 - `model_150k.ckpt` contains the final model as used in the paper.
 - `data/test` and `data/train` contain the USV training and test datasets.

## Model Usage
To load the model:

    import torch
    import torchaudio
    
    device = 'cpu'
    ckpt_path = "model_150k.ckpt"
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    biwavegan = load_model(ckpt)
To generate synthetic USVs:

    batch_size = 16
    z = torch.Tensor(batch_size, biwavegan.latent_dim).uniform_(-1, 1).to(device)
    x = biwavegan.generate(z)
To encode USVs into the latent space:

    x, sample_rate = torchaudio.load("usv.wav")
    # x must be of shape [num_usvs, 1, biwavegan.slice_len]
    if x.size(-1) < biwavegan.slice_len:
        x = torch.nn.functional.pad(x, (0, biwavegan.slice_len - x.size(-1)))
    else:
        x = x[:, :biwavegan.slice_len]
    x = x.unsqueeze(0)
    E_x = biwavegan.encode(x).detach()
To reconstruct a USV:

    recon_x = biwavegan.encode(x).detach()
   
