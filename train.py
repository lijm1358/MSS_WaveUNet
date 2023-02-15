import argparse
import statistics
from typing import Tuple, Dict
import os

import numpy as np

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from data.dataset import MUSDBDataset
from loss.customsdr import ModifiedSDR
from model.waveunet import WaveUNet


class EarlyStopping:
    """Early stopping class for training.

    Note:
        This early stopping class is made for SDR loss. It may not be suitable for other losses.

    Args:
        patience: training will be stopped if the validation loss was not improved more than 'patience'th iteration.
        path: name and path to save best validation model.

    Attributes:
        patience: training will be stopped if the validation loss was not improved more than 'patience'th iteration.
        best_loss: current best validation loss.  initial value is `-inf`.
        counter: count increases when the current validation loss is not improved compare with `best_loss`. will be set to 0 if 
                 current loss has improved.
        early_stop: set to `True` if counter is greater or equal than patience. used for stop validation step.
    """
    def __init__(self, patience=20, path='best.pt'):
        self.patience = patience
        self.best_loss = -np.inf
        self.counter = 0
        self.path = path
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if self.best_loss < val_loss:
            print(f'Validation loss improved({self.best_loss} -> {val_loss})')
            self.counter = 0
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def is_stop(self):
        return self.early_stop

def sdr_loss_mean(y_target, y_pred, loss_fn) -> Tuple[float, list, int]:
    """Calculate the sdr loss of batch.

    Each return value will be used for the average value of accompanies' loss, and median of vocal loss.

    Args:
        y_target: original target batch
        y_pred: prediction of y_target
        loss_fn: loss function(SDR)
    
    Returns:
        float: the sum of accompanies' loss
        list(Tensor): the list of Tensor of calculated vocal loss
        int: the number of accmpanies' loss
    """
    y_target_acc = y_target[:, 0:3]
    y_target_acc = torch.flatten(y_target_acc, end_dim=1)
    y_pred_acc = y_pred[:, 0:3]
    y_pred_acc = torch.flatten(y_pred_acc, end_dim=1)

    y_target_voc = y_target[:, 3]
    y_pred_voc = y_pred[:, 3]

    # remove tensor element only consists of zero.
    mask = (y_target_acc != 0).any(dim=1)   
    y_target_acc = y_target_acc[mask]
    y_pred_acc = y_pred_acc[mask]

    mask = (y_target_voc != 0).any(dim=1)
    y_target_voc = y_target_voc[mask]
    y_pred_voc = y_pred_voc[mask]

    loss_acc = loss_fn(y_pred_acc, y_target_acc)[0].item()
    loss_voc = loss_fn(y_pred_voc, y_target_voc)[1]

    return loss_acc, loss_voc, y_target_acc.shape[0]

def train(dataloader, model, loss_fn, loss_list, optimizer, device):
    """Training step.

    Args:
        dataloader: train dataloader
        model: WaveUNet model
        loss_fn: loss function for training step(MSE)
        loss_list: list to save train losses
        optimizer: optimizer for training step
        device: device to train
    """
    size = len(dataloader.dataset)
    loss_avg = 0
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg += loss.item()

    loss_avg = loss_avg / len(dataloader)
    loss_list.append(loss_avg)
    print(f"train loss : {loss_avg}")


def val(dataloader, model, loss_fn, loss_list, early_stop, device):
    """Validation step.

    Accompany loss will be used for early stopping.

    Args:
        dataloader: validation dataloader
        model: WaveUNet model
        loss_fn: loss function for validation step(SDR)
        loss_list: list to save validation losses
        early_stop: early stopping class instance
        device: device to validation
    """
    model.eval()
    val_loss_acc = 0
    val_loss_voc = []
    data_count = 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            acc_loss, voc_loss, count = sdr_loss_mean(y, pred, loss_fn)
            val_loss_acc += acc_loss
            val_loss_voc += voc_loss
            data_count += count
    val_loss_acc /= data_count
    val_loss_voc = statistics.median(val_loss_voc)
    early_stop(val_loss_acc, model) # only accompanies' sdr will be used for early stopping.
    loss_list.append(val_loss_acc)
    print(f"validation loss : {val_loss_acc} (accompanies loss, mean), {val_loss_voc} (vocal loss, median)\n")

def test(dataloader, model, loss_fn, device):
    """Test step.

    Args:
        dataloader: test dataloader
        model: WaveUNet model
        loss_fn: loss function for test step(SDR)
        device: device to test
    """
    model.eval()
    test_loss_acc = 0
    test_loss_voc = []
    data_count = 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            acc_loss, voc_loss, count = sdr_loss_mean(y, pred, loss_fn)
            test_loss_acc += acc_loss
            test_loss_voc += voc_loss
            data_count += count
    test_loss_acc /= data_count
    test_loss_voc = statistics.median(test_loss_voc)
    print(f"test loss : {test_loss_acc} (accompanies loss, mean), {test_loss_voc} (vocal loss, median)\n")

def load_checkpoint(checkpoint, model, optimizer) -> Dict:
    """Load from saved checkpoint to resume training.

    Args:
        checkpoint: name of saved checkpoint file.
        model: model to load from saved checkpoint
        optimizer: optimizer to load from saved checkpoint
    
    Returns:
        dict: checkpoint state dictionary. model and optimizer's state dictionary will be loaded in this function,
              but `earlystop_bestloss`, `earlystop_counter`, `train_losslist`, `val_losslist`, `epoch` may be used from return value.
    """
    checkpoint = torch.load(checkpoint)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

def main(args):
    filename_bestval = os.path.normpath(args.filename_bestval)
    
    train_loss_list = []
    val_loss_list = []

    train_ds = MUSDBDataset(args.path_train)
    test_ds = MUSDBDataset(args.path_test)
    valid_ds, test_ds = random_split(test_ds, [int(len(test_ds)*0.5), len(test_ds) - int(len(test_ds)*0.5)])

    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size)
    valid_dataloader = DataLoader(valid_ds, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = WaveUNet(n_level=args.n_layers, n_source=args.n_sources).to(device)
    loss_fn = nn.MSELoss()
    test_loss_fn = ModifiedSDR().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999])

    early_stop = EarlyStopping(patience=args.patience, path=filename_bestval)

    epochs = args.max_epoch

    start_epoch=1
    if args.filename_checkpoint is not None:
        checkpoint_dict = load_checkpoint(args.filename_checkpoint, model, optimizer)
        early_stop.best_loss = checkpoint_dict['earlystop_bestloss']
        early_stop.counter = checkpoint_dict['earlystop_counter']

        train_loss_list = checkpoint_dict['train_losslist']
        val_loss_list = checkpoint_dict['val_losslist']

        start_epoch = checkpoint_dict['epoch'] + 1

    for t in range(start_epoch, epochs):
        print(f"epoch : {t}\n---------------------------")
        model.train()
        train(train_dataloader, model, loss_fn, train_loss_list, optimizer, device)
        
        with torch.no_grad():
            val(valid_dataloader, model, test_loss_fn, val_loss_list, early_stop, device)

        if early_stop.is_stop():
            print("Early stop. Loading best model...")
            model.load_state_dict(torch.load(filename_bestval))
            break

        torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'earlystop_bestloss': early_stop.best_loss,
            'earlystop_counter': early_stop.counter,
            'train_losslist': train_loss_list,
            'val_losslist': val_loss_list
        }, os.path.normpath(f'checkpoint/checkpoint_epoch{t}'))

        try:
            os.remove(os.path.normpath(f'checkpoint/checkpoint_epoch{t-1}'))
        except OSError:
            pass

    with torch.no_grad():
        test(test_dataloader, model, test_loss_fn, device)
    torch.save(model.state_dict(), "model.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train WaveUNet')

    parser.add_argument('path_train', type=str, help='the directory where train dataset is stored')
    parser.add_argument('path_test', type=str, help='the directory where test dataset is stored')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=12, help='the number of layers')
    parser.add_argument('--n_sources', type=int, default=4, help='the number of output sources')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--filename_bestval', type=str, default='checkpoint/best.pt', help='path and name of where to save best validation checkpoint')
    parser.add_argument('--filename_checkpoint', type=str, default=None, help='path and name of checkpoint to resume training(Optional)')

    args = parser.parse_args()
    main(args)