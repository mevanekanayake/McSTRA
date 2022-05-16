import pandas as pd
from collections import defaultdict, OrderedDict
import time
import numpy as np
import torch
from utils import evaluate
import random
import logging
import sys
import torch.backends.cudnn as cudnn
import os
from pathlib import Path
import copy


def set_logger(experiments_path, folder_name):
    logger = logging.getLogger()
    filehandler = logging.FileHandler(os.path.join(f'{experiments_path}', f'{folder_name}_logs.log'))
    streamhandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    streamhandler.setFormatter(formatter)
    filehandler.setFormatter(formatter)
    logger.addHandler(streamhandler)
    logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    return logger


def set_seed(seed, deterministic):
    torch.manual_seed(seed)  # pytorch
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.cuda.manual_seed(seed)

    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True


# Run Manager
class RunManager:

    def __init__(self, train_loader, val_loader, experiments_path, folder_name, local_device, ckpt):

        self.experiments_path = experiments_path
        self.folder_name = folder_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epoch_start_time = None
        self.epoch_train_loss = None
        self.epoch_val_loss = None
        self.summary = OrderedDict({"epoch": [],
                                    "training loss": [],
                                    "validation loss": [],
                                    "validation NMSE": [],
                                    "validation PSNR": [],
                                    "validation SSIM": [],
                                    "epoch duration": [],
                                    "validation PD NMSE": [],
                                    "validation PD PSNR": [],
                                    "validation PD SSIM": [],
                                    "validation PDFS NMSE": [],
                                    "validation PDFS PSNR": [],
                                    "validation PDFS SSIM": [],
                                    })

        self.val_log_indices = None
        self.mse_vals = None
        self.target_norms = None
        self.ssim_vals = None
        self.max_vals = None
        self.local_device = local_device

        self.sequences = None

        self.pd_mse_vals = None
        self.pd_target_norms = None
        self.pd_ssim_vals = None
        self.pd_max_vals = None

        self.pdfs_mse_vals = None
        self.pdfs_target_norms = None
        self.pdfs_ssim_vals = None
        self.pdfs_max_vals = None

        self.epoch_count = ckpt['epoch'] if ckpt else 0
        self.avg_epoch_val_loss_min = ckpt['avg_epoch_val_loss_min'] if ckpt else float('inf')
        self.best_model_state = ckpt['best_model_state_dict'] if ckpt else None

    def begin_epoch(self):

        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_train_loss = 0
        self.epoch_val_loss = 0
        self.mse_vals = defaultdict(dict)
        self.target_norms = defaultdict(dict)
        self.ssim_vals = defaultdict(dict)
        self.max_vals = dict()

        self.sequences = defaultdict(dict)

        self.pd_mse_vals = defaultdict(dict)
        self.pd_target_norms = defaultdict(dict)
        self.pd_ssim_vals = defaultdict(dict)
        self.pd_max_vals = dict()

        self.pdfs_mse_vals = defaultdict(dict)
        self.pdfs_target_norms = defaultdict(dict)
        self.pdfs_ssim_vals = defaultdict(dict)
        self.pdfs_max_vals = dict()

    def end_train_step(self, train_loss, size_of_train_batch):
        self.epoch_train_loss += train_loss.item() * size_of_train_batch

    def end_val_step(self, val_logs, size_of_val_batch):
        # check inputs
        for k in (
                "fname",
                "slice_num",
                "max_value",
                "output",
                "target",
                "val_loss",
                "sequence",
        ):
            if k not in val_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by validation_step."
                )
        if val_logs["output"].ndim == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        elif val_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        if val_logs["target"].ndim == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)
        elif val_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")

        for i, fname in enumerate(val_logs["fname"]):
            slice_num = int(val_logs["slice_num"][i])
            maxval = val_logs["max_value"][i].to(self.local_device).numpy()
            output = val_logs["output"][i].to(self.local_device).numpy()
            target = val_logs["target"][i].to(self.local_device).numpy()
            sequence = val_logs["sequence"][i]

            if fname not in self.sequences.keys():
                self.sequences[fname] = sequence

            self.mse_vals[fname][slice_num] = torch.tensor(evaluate.mse(target, output)).view(1)
            self.target_norms[fname][slice_num] = torch.tensor(evaluate.mse(target, np.zeros_like(target))).view(1)
            self.ssim_vals[fname][slice_num] = torch.tensor(
                evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)).view(1)
            self.max_vals[fname] = maxval

            if sequence == "CORPD_FBK":
                self.pd_mse_vals[fname][slice_num] = self.mse_vals[fname][slice_num]
                self.pd_target_norms[fname][slice_num] = self.target_norms[fname][slice_num]
                self.pd_ssim_vals[fname][slice_num] = self.ssim_vals[fname][slice_num]
                self.pd_max_vals[fname] = maxval
            elif sequence == "CORPDFS_FBK":
                self.pdfs_mse_vals[fname][slice_num] = self.mse_vals[fname][slice_num]
                self.pdfs_target_norms[fname][slice_num] = self.target_norms[fname][slice_num]
                self.pdfs_ssim_vals[fname][slice_num] = self.ssim_vals[fname][slice_num]
                self.pdfs_max_vals[fname] = maxval
            else:
                raise RuntimeError(f'Unexpected sequence: {sequence}')

        self.epoch_val_loss += val_logs["val_loss"].item() * size_of_val_batch

    def end_epoch(self, model, optimizer, data_parallel):

        epoch_duration = time.time() - self.epoch_start_time

        cum_val_nmse = 0
        cum_val_psnr = 0
        cum_val_ssim = 0
        total_files = 0

        pd_cum_val_nmse = 0
        pd_cum_val_psnr = 0
        pd_cum_val_ssim = 0
        pd_total_files = 0

        pdfs_cum_val_nmse = 0
        pdfs_cum_val_psnr = 0
        pdfs_cum_val_ssim = 0
        pdfs_total_files = 0

        volume_stats = OrderedDict({"filename": [],
                                    "NMSE": [],
                                    "PSNR": [],
                                    "SSIM": [],
                                    "sequence": [],
                                    "NMSE 18": [],
                                    "PSNR 18": [],
                                    "SSIM 18": [],
                                    })

        for fname in self.mse_vals.keys():
            total_files = total_files + 1
            sequence = self.sequences[fname]
            f_mse_val = torch.mean(torch.cat([v.view(-1) for _, v in self.mse_vals[fname].items()]))
            f_target_norm = torch.mean(torch.cat([v.view(-1) for _, v in self.target_norms[fname].items()]))
            # NMSE of file (or volume)
            f_val_nmse = f_mse_val / f_target_norm
            cum_val_nmse = cum_val_nmse + f_val_nmse
            # PSNR of file (or volume)
            f_val_psnr = 20 * torch.log10(
                torch.tensor(self.max_vals[fname], dtype=f_mse_val.dtype, device=f_mse_val.device)) - 10 * torch.log10(
                f_mse_val)
            cum_val_psnr = cum_val_psnr + f_val_psnr
            # SSIM of file (or volume)
            f_val_ssim = torch.mean(torch.cat([v.view(-1) for _, v in self.ssim_vals[fname].items()]))
            cum_val_ssim = cum_val_ssim + f_val_ssim

            # PD
            if sequence == "CORPD_FBK":
                pd_total_files = pd_total_files + 1
                pd_cum_val_nmse = pd_cum_val_nmse + f_val_nmse
                pd_cum_val_psnr = pd_cum_val_psnr + f_val_psnr
                pd_cum_val_ssim = pd_cum_val_ssim + f_val_ssim
            # PDFS
            elif sequence == "CORPDFS_FBK":
                pdfs_total_files = pdfs_total_files + 1
                pdfs_cum_val_nmse = pdfs_cum_val_nmse + f_val_nmse
                pdfs_cum_val_psnr = pdfs_cum_val_psnr + f_val_psnr
                pdfs_cum_val_ssim = pdfs_cum_val_ssim + f_val_ssim

            # VOLUME-WISE STATS
            volume_stats["filename"].append(fname)
            volume_stats["NMSE"].append(f_val_nmse.item())
            volume_stats["PSNR"].append(f_val_psnr.item())
            volume_stats["SSIM"].append(f_val_ssim.item())
            volume_stats["sequence"].append(sequence)
            volume_stats["NMSE 18"].append((self.mse_vals[fname][18] / self.target_norms[fname][18]).item())
            volume_stats["PSNR 18"].append((20 * torch.log10(torch.tensor(self.max_vals[fname], dtype=f_mse_val.dtype, device=f_mse_val.device)) - 10 * torch.log10(self.mse_vals[fname][18])).item())
            volume_stats["SSIM 18"].append(self.ssim_vals[fname][18].item())

        # EPOCH-WISE STATS
        avg_epoch_val_nmse = cum_val_nmse / total_files
        avg_epoch_val_psnr = cum_val_psnr / total_files
        avg_epoch_val_ssim = cum_val_ssim / total_files
        avg_epoch_train_loss = self.epoch_train_loss / len(self.train_loader.dataset)
        avg_epoch_val_loss = self.epoch_val_loss / len(self.val_loader.dataset)

        if avg_epoch_val_loss < self.avg_epoch_val_loss_min:
            self.avg_epoch_val_loss_min = avg_epoch_val_loss
            pd.DataFrame.from_dict(volume_stats, orient='columns').to_csv(Path(f'{self.experiments_path}', f'{self.folder_name}_best_epoch_stats.csv'), index=False)
            self.best_model_state = copy.deepcopy(model.module.state_dict() if data_parallel else model.state_dict())

        torch.save({'epoch': self.epoch_count,
                    'model_state_dict': model.module.state_dict() if data_parallel else model.state_dict(),
                    'best_model_state_dict': self.best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_epoch_val_loss_min': self.avg_epoch_val_loss_min,
                    }, os.path.join(self.experiments_path, f'{self.folder_name}_model.pth'))

        # EPOCH-WISE STATS
        pd_avg_epoch_val_nmse = pd_cum_val_nmse / pd_total_files if pd_total_files > 0 else torch.tensor(0)
        pd_avg_epoch_val_psnr = pd_cum_val_psnr / pd_total_files if pd_total_files > 0 else torch.tensor(0)
        pd_avg_epoch_val_ssim = pd_cum_val_ssim / pd_total_files if pd_total_files > 0 else torch.tensor(0)

        pdfs_avg_epoch_val_nmse = pdfs_cum_val_nmse / pdfs_total_files if pdfs_total_files > 0 else torch.tensor(0)
        pdfs_avg_epoch_val_psnr = pdfs_cum_val_psnr / pdfs_total_files if pdfs_total_files > 0 else torch.tensor(0)
        pdfs_avg_epoch_val_ssim = pdfs_cum_val_ssim / pdfs_total_files if pdfs_total_files > 0 else torch.tensor(0)

        self.summary["epoch"].append(self.epoch_count)
        self.summary["training loss"].append(avg_epoch_train_loss)
        self.summary["validation loss"].append(avg_epoch_val_loss)
        self.summary["validation NMSE"].append(avg_epoch_val_nmse.item())
        self.summary["validation PSNR"].append(avg_epoch_val_psnr.item())
        self.summary["validation SSIM"].append(avg_epoch_val_ssim.item())
        self.summary["epoch duration"].append(epoch_duration)

        self.summary["validation PD NMSE"].append(pd_avg_epoch_val_nmse.item())
        self.summary["validation PD PSNR"].append(pd_avg_epoch_val_psnr.item())
        self.summary["validation PD SSIM"].append(pd_avg_epoch_val_ssim.item())

        self.summary["validation PDFS NMSE"].append(pdfs_avg_epoch_val_nmse.item())
        self.summary["validation PDFS PSNR"].append(pdfs_avg_epoch_val_psnr.item())
        self.summary["validation PDFS SSIM"].append(pdfs_avg_epoch_val_ssim.item())

        pd.DataFrame.from_dict(self.summary, orient='columns').to_csv(Path(os.path.join(f'{self.experiments_path}', f'{self.folder_name}_summary.csv')), index=False)
