import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from utils_new.evaluate import SSIMLoss
from utils_new.data import SliceDataset
from utils_new.data import fetch_dir_new
from utils_new.mask import create_mask_for_mask_type, worker_init_fn
from utils_new.transform_castra_final_ht7 import CASTRADataTransform, make_ht_masks
from utils_new.manager import RunManager, set_seed, set_logger

from tqdm import tqdm
from datetime import datetime
import platform
import os
import argparse
from pathlib import Path
import random

from models.castra_final_ht7 import CASTRA

from packaging import version

if version.parse(torch.__version__) >= version.parse("1.7.0"):
    from utils_new.fourier import fft2c_new as fft2c
    from utils_new.fourier import ifft2c_new as ifft2c
else:
    from utils_new.fourier import fft2c_old as fft2c
    from utils_new.fourier import ifft2c_old as ifft2c


def train_castra_():
    # SET ARGUMENTS
    parser = argparse.ArgumentParser()

    # DATA ARGS
    parser.add_argument("--acceleration_factors", type=list, default=[4], help="Acceleration factors for the k-space undersampling")
    parser.add_argument("--center_fractions", type=list, default=[0.08], help="Fractions of center frequencies preserved")
    parser.add_argument("--tvsr", type=float, default=.001, help="Fraction of data volumes used for training")
    parser.add_argument("--vvsr", type=float, default=.005, help="Fraction of data volumes used for validation")
    parser.add_argument("--mask_type", type=str, choices=("random", "equispaced"), default="equispaced", help="Type of k-space mask")
    parser.add_argument("--challenge", choices=("singlecoil", "multicoil"), default="singlecoil", type=str, help="Which challenge to preprocess for")
    parser.add_argument("--anatomy", choices=("knee", "brain"), default="knee", type=str, help="Which anatomy to preprocess for")

    # TRAIN ARGS
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training and validation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for model training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--data_parallel", type=str, default=None, help="Whether to perform Data parallelism")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes for data loading")
    parser.add_argument("--checkpoint", type=str, help="Continue trainings from checkpoint")

    # MODEL ARGS
    parser.add_argument("--num_unets", type=int, default=5, help="Number of U-Nets in cascade")
    parser.add_argument("--img_size", type=int, default=320, help="Size of the input images")
    parser.add_argument("--patch_size", type=int, default=4, help="the intiial patch size to which the input image is divided")
    parser.add_argument("--in_chans", type=int, default=2, help="Number of channels in the input image ")
    parser.add_argument("--out_chans", type=int, default=2, help="Number of channels in the output image ")
    parser.add_argument("--embed_dims", type=str, default='48,96,48', help="Embedding dimension of the heads, cascade, tail")
    parser.add_argument("--depths", type=str, default='2,2,2,2', help="Depth of each layer of the U-Net")
    parser.add_argument("--num_heads", type=str, default='3,6,12,24', help="Number of heads at each layer of the U-Net")
    parser.add_argument("--window_size", type=int, default=5, help="Number of U-Nets in cascade")
    parser.add_argument("--mlp_ratio", type=int, default=4, help="Number of MLP hidden layer nodes")
    parser.add_argument("--drop_rate", type=float, default=0., help="DropOut rate")
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="Drop Path rate")
    parser.add_argument("--num_hts", type=int, default=2, help="Number of heads and tails")
    parser.add_argument("--ape", type=str, default=True, help="Absolute Positional Embedding")

    args = parser.parse_args()
    ckpt = torch.load(Path(args.checkpoint), map_location='cpu') if args.checkpoint else None

    # SET SEED and CUDA BENCHMARK/DETERMINISTIC
    set_seed(seed=42, deterministic=1)  # set deterministic to 1 if the input size remains same

    # SET PATHS
    data_path = fetch_dir_new(node=platform.node()[:2], what_path=args.anatomy, data_config_file="paths.json")
    folder_name = "Experiment_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experiments_path = fetch_dir_new(node=platform.node()[:2], what_path='experiments', data_config_file="paths.json", folder=folder_name)
    if not os.path.isdir(experiments_path):
        os.makedirs(experiments_path)  # create the experiments_path if it does not exist

    # LOG ARGS, PATHS
    logger = set_logger(experiments_path, folder_name)
    for entry in vars(args):
        logger.info(f'{entry}: {vars(args)[entry]}')
    logger.info(f'data_path = {str(data_path)}')
    logger.info(f'experiments_path = {str(experiments_path)}')

    # LOAD MODEL
    model = CASTRA(args)
    model.load_state_dict(ckpt['model_state_dict']) if args.checkpoint else None
    logger.info(f'No. of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # SET GPUS
    if args.data_parallel:
        device_ids = args.data_parallel.split(',')
        args.device = f'cuda:{device_ids[0]}'
        device_ids = [int(device_id) for device_id in device_ids]
        model = nn.DataParallel(model, device_ids=device_ids)
        logger.info(f'num GPUs available: {torch.cuda.device_count()} | num GPUs using: {len(device_ids)}')

    logger.info(f'Device: {torch.cuda.get_device_name(args.device)}') if torch.cuda.device_count() > 0 else logger.info(f'Device: CPU')
    model.to(args.device)

    # CREATE MASK
    print('Creating mask ...')
    mask = create_mask_for_mask_type(mask_type_str=args.mask_type,
                                     center_fractions=args.center_fractions,
                                     accelerations=args.acceleration_factors)

    # LOAD TRAINING DATA
    # use random masks for train transform, fixed masks for val transform
    train_transform = CASTRADataTransform(which_challenge=args.challenge,
                                          mask_func=mask,
                                          use_seed=False)
    print('Gathering training data ...')
    train_set = SliceDataset(root=data_path / f"{args.challenge}_train",
                             transform=train_transform,
                             sample_rate=None,
                             volume_sample_rate=args.tvsr,
                             challenge=args.challenge)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              worker_init_fn=worker_init_fn,
                              shuffle=True,
                              pin_memory=True)
    logger.info(f'Training set gathered: No. of volumes: {len(set([fl[0] for fl in train_loader.dataset.examples]))} | No. of slices: {len(train_set)}')

    # LOAD VALIDATION DATA
    val_transform = CASTRADataTransform(which_challenge=args.challenge,
                                        mask_func=mask)
    print('Gathering validation data ...')
    val_set = SliceDataset(root=data_path / f"{args.challenge}_val",
                           transform=val_transform,
                           sample_rate=None,
                           volume_sample_rate=args.vvsr,
                           challenge=args.challenge)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            worker_init_fn=worker_init_fn,
                            shuffle=False,
                            pin_memory=True)
    logger.info(f'Validation set gathered: No. of volumes: {len(set([fl[0] for fl in val_loader.dataset.examples]))} | No. of slices: {len(val_set)}')

    # LOSS FUNCTION
    logger.info(f'Loss function: SSIM loss, L1 loss, L2 loss')  # change this line manually if you change the loss function
    # loss_SSIM = SSIMLoss().to(args.device)
    loss_L1 = nn.L1Loss()
    # loss_L2 = nn.MSELoss()

    # SET OPTIMIZER & SCHEDULER
    logger.info(f'Optimizer: RMSprop')  # change this line manually if you change the optimizer
    optimizer = torch.optim.RMSprop(params=model.parameters(),
                                    lr=args.lr,
                                    weight_decay=0.)
    optimizer.load_state_dict(ckpt['optimizer_state_dict']) if args.checkpoint else None

    # INITIALIZE RUN MANAGER
    m = RunManager(train_loader, val_loader, experiments_path, folder_name, 'cpu', ckpt)

    # HT MASKS
    ht_masks = make_ht_masks(args.num_hts, args.center_fractions)

    # LOSS COMPUTATIONS
    lc = args.num_unets
    alphas = [(i + 1) / (lc * (lc + 1) / 2) for i in range(lc)]

    # LOOP
    for _ in range(args.num_epochs):
        # BEGIN EPOCH
        m.begin_epoch()

        # BEGIN TRAINING LOOP
        model.train()
        with tqdm(train_loader, unit="batch") as train_epoch:

            for train_batch in train_epoch:
                train_epoch.set_description(f"Epoch {m.epoch_count} [Training]")

                target1 = train_batch[1].to(args.device)
                kspace_und = train_batch[2].to(args.device)
                mask = train_batch[3].to(args.device)
                target2 = train_batch[10].to(args.device)
                masked_psf = train_batch[11].to(args.device)
                kspace_ori = train_batch[12].to(args.device)

                head_ips = []
                head_targets = []

                import matplotlib.pyplot as plt
                plt.imshow(target1[0,0].detach().cpu(), cmap='gray')
                plt.show()

                # condition input
                for ht in range(args.num_hts):
                    kspace = kspace_und * ht_masks[ht].to(args.device).unsqueeze(0).unsqueeze(0)
                    image = ifft2c(kspace.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                    head_ips.append(image)

                # condition target
                for ht in range(args.num_hts):
                    kspace = kspace_ori * ht_masks[ht].to(args.device).unsqueeze(0).unsqueeze(0)
                    target = ifft2c(kspace.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                    head_targets.append(target)

                # import matplotlib.pyplot as plt
                # from utils.transform import complex_abs
                # plot_list = [complex_abs(head_ips[0][1].detach().cpu().permute(1, 2, 0)),
                #              complex_abs(head_targets[0][1].detach().cpu().permute(1, 2, 0)),
                #              complex_abs(head_ips[1][1].detach().cpu().permute(1, 2, 0)),
                #              complex_abs(head_targets[1][1].detach().cpu().permute(1, 2, 0)),
                #              complex_abs(head_ips[2][1].detach().cpu().permute(1, 2, 0)),
                #              complex_abs(head_targets[2][1].detach().cpu().permute(1, 2, 0))
                #              ]
                #
                # for i, ht_mask in enumerate(plot_list):
                #     ax = plt.imshow(ht_mask.detach().cpu(), cmap='gray')
                #     ax.axes.get_xaxis().set_visible(False)
                #     ax.axes.get_yaxis().set_visible(False)
                #     ax = plt.show()
                #     # from pathlib import Path
                #     # figure_path = Path('D:\\TMI\\figures')
                #     # plt.savefig(f'{figure_path}/ht7_{i}.png', format='png', bbox_inches='tight', dpi=600)

                optimizer.zero_grad()
                outputs = model(head_ips, kspace_und, mask, masked_psf)

                # head losses
                head_loss = torch.tensor(0., device=args.device)
                for i in range(args.num_hts):
                    alpha = 1/args.num_hts
                    train_loss_a = loss_L1(outputs[i], head_targets[i])
                    head_loss += alpha * train_loss_a

                # cascade losses
                cascade_loss = torch.tensor(0., device=args.device)
                for i in range(args.num_unets):
                    alpha = alphas[i]
                    train_loss_a = loss_L1(outputs[i+args.num_hts], target2)
                    cascade_loss += alpha * train_loss_a

                # tail loss
                tail_loss = loss_L1(outputs[args.num_hts+args.num_unets], target1)

                # total loss
                train_loss = (head_loss + cascade_loss + tail_loss)/3

                train_loss.backward()
                optimizer.step()

                # END TRAINING STEP
                train_epoch.set_postfix(train_loss=train_loss.detach().item())
                m.end_train_step(train_loss.detach(), train_batch[0].shape[0])

        # BEGIN VALIDATION LOOP
        model.eval()
        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as val_epoch:
                for val_batch in val_epoch:
                    val_epoch.set_description(f"Epoch {m.epoch_count} [Validation]")

                    target1 = val_batch[1].to(args.device)
                    kspace_und = val_batch[2].to(args.device)
                    mask = val_batch[3].to(args.device)
                    fname = val_batch[4]
                    slice_num = val_batch[5]
                    max_value = val_batch[6].to(args.device)
                    sequence = val_batch[9]
                    target2 = val_batch[10].to(args.device)
                    masked_psf = val_batch[11].to(args.device)
                    kspace_ori = val_batch[12].to(args.device)

                    head_ips = []
                    head_targets = []

                    # condition input
                    for ht in range(args.num_hts):
                        kspace = kspace_und * ht_masks[ht].to(args.device).unsqueeze(0).unsqueeze(0)
                        image = ifft2c(kspace.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                        head_ips.append(image)

                    # condition target
                    for ht in range(args.num_hts):
                        kspace = kspace_ori * ht_masks[ht].to(args.device).unsqueeze(0).unsqueeze(0)
                        target = ifft2c(kspace.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                        head_targets.append(target)

                    optimizer.zero_grad()
                    outputs = model(head_ips, kspace_und, mask, masked_psf)

                    # head losses
                    head_loss = torch.tensor(0., device=args.device)
                    for i in range(args.num_hts):
                        alpha = 1 / args.num_hts
                        train_loss_a = loss_L1(outputs[i], head_targets[i])
                        head_loss += alpha * train_loss_a

                    # cascade losses
                    cascade_loss = torch.tensor(0., device=args.device)
                    for i in range(args.num_unets):
                        alpha = alphas[i]
                        train_loss_a = loss_L1(outputs[i + args.num_hts], target2)
                        cascade_loss += alpha * train_loss_a

                    # tail loss
                    tail_loss = loss_L1(outputs[args.num_hts + args.num_unets], target1)

                    # total loss
                    val_loss = (head_loss + cascade_loss + tail_loss) / 3

                    # END VALIDATION STEP
                    val_epoch.set_postfix(val_loss=val_loss)
                    m.end_val_step({
                        "fname": fname,
                        "slice_num": slice_num,
                        "max_value": max_value,
                        "output": outputs[-1].squeeze(1).to('cpu'),
                        "target": target1.squeeze(1).to('cpu'),
                        "val_loss": val_loss.to('cpu'),
                        "sequence": sequence,
                    }, val_batch[0].shape[0])

        # END EPOCH
        m.end_epoch(model, optimizer, args.data_parallel)


if __name__ == '__main__':
    train_castra_()
    print('Done training!')
