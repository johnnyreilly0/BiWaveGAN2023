import torch
import model
import torch.optim as optim
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('logdir')
    parser.add_argument('--slice_len', default=32768, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--beta2', default=0.9, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_iters', default=100000, type=int)
    parser.add_argument('--lambda_gp', default=10, type=float)
    parser.add_argument('--d_iters', default=5, type=int)
    parser.add_argument('--latent_dim', default=32, type=int)
    parser.add_argument('--model_size', default=32, type=int)
    parser.add_argument('--discrim_filters', default=512, type=int)
    parser.add_argument('--z_discrim_depth', default=2, type=int)
    parser.add_argument('--joint_discrim_depth', default=3, type=int)
    parser.add_argument('--phaseshuffle_rad', default=2, type=int)
    parser.add_argument('--sample_rate', default=250000, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--val_size', default=1000, type=int)
    parser.add_argument('--recon_loss_weight', default=0, type=float)
    return parser.parse_args()


def get_next_batch(train_iter, loader, device):
    try:
        batch = next(train_iter).to(device)
    except StopIteration:
        train_iter = iter(loader)
        batch = next(train_iter).to(device)
    return batch, train_iter


def save_state(args, model, optimEG, optimD, it, path):
    state = {
            "slice len": args.slice_len,
            "latent dim": args.latent_dim,
            "model size": args.model_size,
            "phaseshuffle rad": args.phaseshuffle_rad,
            "disrim filters": args.discrim_filters,
            "z discrim depth": args.z_discrim_depth,
            "joint discrim depth": args.joint_discrim_depth,
            "G state_dict": model.G.state_dict(),
            "E state_dict": model.E.state_dict(),
            "D state_dict": model.D.state_dict(),
            "EG optimiser": optimEG.state_dict(),
            "D optimiser": optimD.state_dict(),
            "iter": it,
        }
    torch.save(state, path)


def save_run_data(EG_losses, D_losses, recon_error_list, path):
    run_data = {
        "EG losses": EG_losses,
        "D losses": D_losses,
        "val recon errors": recon_error_list
    }
    torch.save(run_data, path)


def load_model(ckpt, device):
    bwg = model.BiWaveGAN(
        slice_len=32768,
        latent_dim=ckpt['latent dim'],
        model_size=ckpt['model size'],
        discrim_filters=ckpt['disrim filters'],
        z_discrim_depth=ckpt['z discrim depth'],
        joint_discrim_depth=ckpt['joint discrim depth'],
        phaseshuffle_rad=ckpt['phaseshuffle rad'],
        device=device)
    bwg.G.load_state_dict(ckpt['G state_dict'])  # , strict=strict)
    bwg.E.load_state_dict(ckpt['E state_dict'])
    bwg.D.load_state_dict(ckpt['D state_dict'])
    bwg.eval()

    return bwg

def load_optimisers(bwg, ckpt):
    optimEG = optim.Adam(list(bwg.G.parameters()) + list(bwg.E.parameters()), lr=ckpt['learning rate'],
    betas = (0.5, 0.9))
    optimD = optim.Adam(bwg.D.parameters(), lr=ckpt['learning rate'], betas=(0.5, 0.9))
    optimEG.load_state_dict(ckpt['EG optimiser'])
    optimD.load_state_dict(ckpt['D optimiser'])
    return optimEG, optimD
