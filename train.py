import logging
import multiprocessing
import os
import time
import requests

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import modules.commons as commons
import utils
from data_utils import TextAudioCollate, TextAudioSpeakerLoader
from sovitsmodels import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from mi_estimators import CLUB
from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

"""
MI code modified from: https://github.com/Wendison/VQMIVC/blob/main/mi_estimators.py
"""


def download_pretrained_model(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}")

def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    hps = utils.get_hparams()

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = hps.train.port


    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    
    # for pytorch on win, backend use gloo    
    dist.init_process_group(backend=  'gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    if rank == 0:
        download_pretrained_model("https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/D_0.pth", os.path.join(hps.model_dir, "D_0.pth"))
        download_pretrained_model("https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/G_0.pth", os.path.join(hps.model_dir, "G_0.pth"))


    collate_fn = TextAudioCollate()
    all_in_mem = hps.train.all_in_mem   # If you have enough memory, turn on this option to avoid disk IO and speed up training.
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps, all_in_mem=all_in_mem)
    num_workers = 5 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()
    if all_in_mem:
        num_workers = 0
    train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
                              batch_size=hps.train.batch_size, collate_fn=collate_fn)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps, all_in_mem=all_in_mem,vol_aug = False)
        eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
                                 batch_size=1, pin_memory=False,
                                 drop_last=False, collate_fn=collate_fn)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    

    cs_mi_net = CLUB(192, 192, 512).cuda(rank)
    ps_mi_net = CLUB(192, 1, 512).cuda(rank)
    cp_mi_net = CLUB(192, 1, 512).cuda(rank)

    optimizer_cs_mi_net = torch.optim.AdamW(cs_mi_net.parameters(),
                                             lr=3e-4)
    optimizer_ps_mi_net = torch.optim.AdamW(ps_mi_net.parameters(),
                                             lr=3e-4)
    optimizer_cp_mi_net = torch.optim.AdamW(cp_mi_net.parameters(),
                                             lr=3e-4)
    
    #3e-4
    
    #Load pre-trained models
    try:
        _, _, _, epoch_str = utils.load_checkpoint(os.path.join(hps.model_dir, "G_0.pth"), net_g, optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(os.path.join(hps.model_dir, "D_0.pth"), net_d, optim_d)
        global_step = 0  # Reset global step when starting from pre-trained model
        epoch_str = 1  # Start from epoch 1
        
    except Exception as e:
        print(f"Failed to load pre-trained models: {e}")
        epoch_str = 1
        global_step = 0

    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank])


    skip_optimizer = False
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                                                   optim_g, skip_optimizer)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
                                                   optim_d, skip_optimizer)
        epoch_str = max(epoch_str, 1)
        name=utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        global_step=int(name[name.rfind("_")+1:name.rfind(".")])+1
        #global_step = (epoch_str - 1) * len(train_loader)
    except Exception:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    warmup_epoch = hps.train.warmup_epochs
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        # set up warm-up learning rate
        if epoch <= warmup_epoch:
            for param_group in optim_g.param_groups:
                param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
            for param_group in optim_d.param_groups:
                param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
        # training
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d, cs_mi_net, ps_mi_net, cp_mi_net],
                           [optim_g, optim_d, optimizer_cs_mi_net, optimizer_ps_mi_net, optimizer_cp_mi_net],
                           [scheduler_g, scheduler_d], scaler,
                           [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d, cs_mi_net, ps_mi_net, cp_mi_net],
                           [optim_g, optim_d, optimizer_cs_mi_net, optimizer_ps_mi_net, optimizer_cp_mi_net],
                           [scheduler_g, scheduler_d], scaler,
                           [train_loader, None], None, None)
        # update learning rate
        scheduler_g.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d, cs_mi_net, ps_mi_net, cp_mi_net = nets
    optim_g, optim_d, optimizer_cs_mi_net, optimizer_ps_mi_net, optimizer_cp_mi_net = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers
    
    half_type = torch.bfloat16 if hps.train.half_type=="bf16" else torch.float16

    global global_step

    net_g.train()
    net_d.train()
    cs_mi_net.train()
    ps_mi_net.train()
    cp_mi_net.train()

    for batch_idx, items in enumerate(train_loader):

        c, w, we, f0, spec, y, spk, lengths, uv, volume, p = items
        g = spk.cuda(rank, non_blocking=True)
        spec, y = spec.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
        c = c.cuda(rank, non_blocking=True)
        w = w.cuda(rank, non_blocking=True)
        we = we.cuda(rank, non_blocking=True)
        f0 = f0.cuda(rank, non_blocking=True)
        uv = uv.cuda(rank, non_blocking=True)
        lengths = lengths.cuda(rank, non_blocking=True)
        p = p.cuda(rank, non_blocking=True)
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax)

        #train mi nets
        first_forward(scaler, net_g, cs_mi_net, ps_mi_net, cp_mi_net,  optimizer_cs_mi_net, optimizer_ps_mi_net, optimizer_cp_mi_net, c, w, we, f0, uv, p, spec, g, lengths ,volume, mel)

        # train gen net
        loss_disc, loss_gen, loss_fm, loss_mel, loss_kl, loss_lf0, mi_loss, loss_gen_all, loss_disc_all, grad_norm_d, grad_norm_g, y_mel, y_hat_mel, lf0, pred_lf0, norm_lf0 = second_forward(scaler, half_type, hps, net_g, net_d, optim_g, optim_d, cs_mi_net, ps_mi_net, cp_mi_net,  c, w, we, f0, uv, p, spec, g, y, lengths ,volume, mel)
        
        # log 
        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl, mi_loss]
                reference_loss = sum(losses)
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info(f"Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}, reference_loss: {reference_loss}")

                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr,
                               "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl,
                                    "loss/g/lf0": loss_lf0, "loss/g/mi" : mi_loss})

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy())
                }

                if net_g.module.use_automatic_f0_prediction:
                    image_dict.update({
                        "all/lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                              pred_lf0[0, 0, :].detach().cpu().numpy()),
                        "all/norm_lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                                   norm_lf0[0, 0, :].detach().cpu().numpy())
                    })

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict
                )

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)

        global_step += 1

    if rank == 0:
        global start_time
        now = time.time()
        duration = format(now - start_time, '.2f')
        logger.info(f'====> Epoch: {epoch}, cost {duration} s')
        start_time = now

def first_forward(scaler, net_g, cs_mi_net, ps_mi_net, cp_mi_net,  optimizer_cs_mi_net, optimizer_ps_mi_net, optimizer_cp_mi_net, c, w, we, f0, uv, p, spec, g, lengths ,volume, mel):
            
    with torch.no_grad():
         # First forward pass for MI loss calc (without gradients)
        y_hat, ids_slice, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0, z_ptemp, gg, x, pros, fused = net_g(c, w, we, f0, uv, p, mel, spec, g=g, c_lengths=lengths,
                                                                                    spec_lengths=lengths, vol=volume)
    # gg # dim: 768
    # pred_lf0 # dim: 192
    # lf0 dim: 1
    # fused # dim: 192
    # z # dim: 192
    # z_ptemp # dim: 192
    # z_p # dim: 192    
    # pp dim: 2

    z_ptemp = z_ptemp.detach()
    z_p = z_p.detach()
    z = z.detach()
    gg = gg.detach()
    lf0 = lf0.detach()
    pred_lf0 = pred_lf0.detach()
    pros = pros.detach()
    fused = fused.detach()
    x = x.detach()

    optimizer_cs_mi_net.zero_grad()
    optimizer_ps_mi_net.zero_grad()
    optimizer_cp_mi_net.zero_grad()

    # MI between content and pros
    lld_cs_loss = -cs_mi_net.loglikeli(fused, pros)
    scaler.scale(lld_cs_loss).backward()
    scaler.unscale_(optimizer_cs_mi_net)
    scaler.step(optimizer_cs_mi_net) 

    # # # MI between content and lf0
    lld_ps_loss = -ps_mi_net.loglikeli(fused, lf0)
    scaler.scale(lld_ps_loss).backward()
    scaler.unscale_(optimizer_ps_mi_net)
    scaler.step(optimizer_ps_mi_net)

    # # # MI between prosody and lf0
    lld_cp_loss = -cp_mi_net.loglikeli(pros, lf0)
    scaler.scale(lld_cp_loss).backward()
    scaler.unscale_(optimizer_cp_mi_net)
    scaler.step(optimizer_cp_mi_net)

    return 

def second_forward(scaler, half_type, hps, net_g, net_d, optim_g, optim_d, cs_mi_net, ps_mi_net, cp_mi_net,  c, w, we, f0, uv, p, spec, g, y, lengths ,volume, mel):

    with autocast(enabled=hps.train.fp16_run, dtype=half_type):
        y_hat, ids_slice, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0, z_ptemp, gg, x, pros, fused = net_g(c, w, we, f0, uv, p, mel, spec, g=g, c_lengths=lengths,
                                                                                    spec_lengths=lengths, vol=volume)

        y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
        y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
        )

        y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

        # Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

        with autocast(enabled=False, dtype=half_type):
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc
        
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run, dtype=half_type):
        # Generator
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
        with autocast(enabled=False, dtype=half_type):

            loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_lf0 = F.mse_loss(pred_lf0, lf0) if net_g.module.use_automatic_f0_prediction else 0

            # # MI estimation 
            mi_cs_loss = cs_mi_net.mi_est(fused, pros) * 0.01
            mi_ps_loss = ps_mi_net.mi_est(fused, lf0) * 0.01
            mi_cp_loss = cp_mi_net.mi_est(pros, lf0) * 0.01

            mi_loss = mi_cs_loss + mi_ps_loss + mi_cp_loss

            # 1 / (mi_cs_loss + 1e-8)

            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0 + mi_loss

    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    return loss_disc, loss_gen, loss_fm, loss_mel, loss_kl, loss_lf0, mi_loss, loss_gen_all, loss_disc_all, grad_norm_d, grad_norm_g, y_mel, y_hat_mel, lf0, pred_lf0, norm_lf0

   
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            c, w, we, f0, spec, y, spk, _, uv,volume, p = items
            g = spk[:1].cuda(0)
            spec, y = spec[:1].cuda(0), y[:1].cuda(0)
            c = c[:1].cuda(0)
            w = w[:1].cuda(0)
            we = we[:1].cuda(0)
            f0 = f0[:1].cuda(0)
            uv= uv[:1].cuda(0)
            p = p[:1].cuda(0)
            if volume is not None:
                volume = volume[:1].cuda(0)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)
            
            y_hat,_ = generator.module.infer(c, w, we, f0, uv, p, mel, g=g,vol = volume)

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            audio_dict.update({
                f"gen/audio_{batch_idx}": y_hat[0],
                f"gt/audio_{batch_idx}": y[0]
            })
        image_dict.update({
            "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
            "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
        })
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
    main()