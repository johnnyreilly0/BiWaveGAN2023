import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import BiWaveGAN
import utils.data_utils
import utils.train_utils
import datetime

# TODO: make sure only one random batch (what size) of real inputs gets sampled for reconstruction visualisation
# TODO: sort out number of data plotted reconstructed etc.

args = utils.train_utils.get_args()

# training params
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# logging and plotting params
ITERS_PER_VALIDATE = 100
ITERS_PER_CHECKPOINT = 10000
N_FFT = 512
HOP_LENGTH = 64
spectrogram = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH)

torch.manual_seed(args.seed)

# split into train and validation sets
train_dataset = utils.data_utils.WAVDataset(args.datadir, sample_rate=args.sample_rate, slice_len=args.slice_len)
if args.val_size:
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,
                                                               [len(train_dataset) - args.val_size, args.val_size],
                                                               torch.Generator().manual_seed(args.seed))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
if args.val_size:
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
train_iter = iter(train_loader)

# create models on device
model = BiWaveGAN(
    slice_len=args.slice_len, latent_dim=args.latent_dim, model_size=args.model_size,
    phaseshuffle_rad=args.phaseshuffle_rad, discrim_filters=args.discrim_filters, z_discrim_depth=args.z_discrim_depth,
    joint_discrim_depth=args.joint_discrim_depth, device=device
)

# create optimisers
optimEG = optim.Adam(list(model.G.parameters()) + list(model.E.parameters()), lr=args.learning_rate,
                     betas=(args.beta1, args.beta2))
optimD = optim.Adam(model.D.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))

# for plotting and logging
fixed_noise = torch.Tensor(16, args.latent_dim).uniform_(-1, 1).to(device)
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join(args.logdir, now)
writer = SummaryWriter(logdir)
EG_losses = []
D_losses = []
recon_error_list = []

# Save args to file
with open(os.path.join(logdir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(arg) + ': ' + str(value) for arg, value in sorted(vars(args).items(), key=lambda x: x[0])]))

model.train()

print(f"Training started at {now}, logdir: {logdir}")
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"log directory: {logdir}")

# make all models trainable
for it in range(args.n_iters):
    for p in model.D.parameters():
        p.requires_grad = True
    for p in model.G.parameters():
        p.requires_grad = True
    for p in model.E.parameters():
        p.requires_grad = True

    one = torch.tensor(1, dtype=torch.float, device=device)
    neg_one = one * -1

    # D iterations
    for _ in range(1, args.d_iters + 1):
        model.D.zero_grad()
        # grab next batch
        real, train_iter = utils.train_utils.get_next_batch(train_iter, train_loader, device)
        cur_batch_size = real.shape[0]

        # generate real and fake latent vectors
        z_real = model.E(real)
        z_fake = torch.Tensor(cur_batch_size, 1, args.latent_dim).uniform_(-1, 1).to(device)
        fake = model.G(z_fake)

        # compute D loss and update
        D_real = model.D(real, z_real).reshape(-1)
        D_fake = model.D(fake, z_fake).reshape(-1)
        gp = model.D.gradient_penalty(real, z_real, fake, z_fake, device=device)
        loss_D = -1 * (torch.mean(D_real) - torch.mean(D_fake)) + args.lambda_gp * gp
        loss_D.backward(retain_graph=True)
        optimD.step()

    # stop D from updating
    for p in model.G.parameters():
        p.requires_grad = True
    for p in model.E.parameters():
        p.requires_grad = True
    for p in model.D.parameters():
        p.requires_grad = False

    # G iteration
    model.E.zero_grad()
    model.G.zero_grad()
    # grab next batch
    real, train_iter = utils.train_utils.get_next_batch(train_iter, train_loader, device)
    cur_batch_size = real.shape[0]

    # generate real and fake latent vectors
    z_real = model.E(real)
    z_fake = torch.Tensor(cur_batch_size, args.latent_dim).uniform_(-1, 1).to(device)
    fake = model.G(z_fake)

    # compute encoder-generator loss and update
    D_real = model.D(real, z_real).reshape(-1)
    D_fake = model.D(fake, z_fake).reshape(-1)
    loss_EG = torch.mean(D_real) - torch.mean(D_fake)
    loss_recon = F.mse_loss(real, model.reconstruct(real))
    if args.recon_loss_weight:
        loss_EG += args.recon_loss_weight * loss_recon
    loss_EG.backward()
    optimEG.step()

    EG_losses.append(loss_EG.item())
    D_losses.append(loss_D.item())
    # log losses to tensorboard
    writer.add_text(
        "Progress", f"Batch {it}/{args.n_iters}" +
                    f"EG loss: {loss_EG:.4f}, D loss: {loss_D:.4f}",
        global_step=it
    )
    writer.add_scalar("loss/EG", loss_EG, global_step=it)
    writer.add_scalar("loss/reconstruction", loss_recon, global_step=it)
    writer.add_scalar("loss/D", loss_D, global_step=it)

    if args.val_size and it % ITERS_PER_VALIDATE == 0:
        # update tensorboard log
        with torch.no_grad():
            recon = model.reconstruct(real).to('cpu')
            real = real.to('cpu')
            fake = model.G(fixed_noise).to('cpu').detach()
            real_specs = spectrogram(real)
            recon_specs = spectrogram(recon)
            fake_specs = spectrogram(fake)
            writer.add_images("Train/real spectrograms", real_specs, global_step=it)
            writer.add_images("Train/reconstructed spectrograms", recon_specs, global_step=it)
            writer.add_images("Train/fake spectrograms", fake_specs, global_step=it)
            writer.add_audio("Train/real audio", real.flatten(), global_step=it, sample_rate=args.sample_rate)
            writer.add_audio("Train/reconstructed audio", recon.flatten(), global_step=it,
                             sample_rate=args.sample_rate)
            writer.add_audio("Train/fake audio", fake.flatten(), global_step=it, sample_rate=args.sample_rate)

            # calculate MSE reconstruction error on val set
            recon_error = 0
            for x in val_loader:
                recon_x = model.reconstruct(x.to(device)).to('cpu')
                recon_error += F.mse_loss(x, recon_x, reduction='sum')
            recon_error /= len(val_dataset)
            recon_error_list.append(recon_error)
        writer.add_scalar("Validation/reconstruction error", recon_error, global_step=it)
        model.train()

    if it % ITERS_PER_CHECKPOINT == 0 and it not in [0, args.n_iters]:
        utils.train_utils.save_state(args, model, optimEG, optimD, it, os.path.join(logdir, f"model.ckpt"))
        utils.train_utils.save_run_data(EG_losses, D_losses, recon_error_list, os.path.join(logdir, f"data.ckpt"))

        writer.add_text("model checkpoint", f"model saved after iter {it}", global_step=it)

# save final model state.
utils.train_utils.save_state(args, model, optimEG, optimD, it, os.path.join(logdir, f"final_model_it{it}.ckpt"))
utils.train_utils.save_run_data(EG_losses, D_losses, recon_error_list, os.path.join(logdir, f"data_it{it}.ckpt"))
