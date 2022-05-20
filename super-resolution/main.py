from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SRNet
from data import get_training_set, get_test_set

import wandb


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--wandb', type=str, help='Wandb experiment name. Il not specified wandb is not used')
opt = parser.parse_args()

if opt.wandb:
    wandb.login()

    # WandB – Initialize a new run
    run = wandb.init(project="super-resolution", name=opt.wandb)
    wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config          # Initialize config
    config.upscale_factor = opt.upscale_factor     # super resolution upscale factor
    config.batch_size = opt.batchSize          # input batch size for training (default: 64)
    config.test_batch_size = opt.testBatchSize    # input batch size for testing (default: 1000)
    config.epochs =  opt.nEpochs            # number of epochs to train (default: 10)
    config.lr = opt.lr               # learning rate (default: 0.01)
    config.cuda = opt.cuda         # enables CUDA training
    config.threads = opt.threads    # number of threads for data loader to use
    config.seed = opt.seed               # random seed (default: 42)



print(f"TRAINING WITH: {opt}")

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = SRNet(upscale_factor=opt.upscale_factor).to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def train(epoch):
    avg_psnr = 0
    avg_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)
        prediction = model(input)

        optimizer.zero_grad()
        loss = criterion(prediction, target)
        avg_loss += loss.item()

        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr

        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    avg_psnr /= len(training_data_loader)
    avg_loss /= len(training_data_loader)
    print("===> Epoch {} Complete: Avg. Train Loss: {:.4f} | Avg. Train PSNR: {:.4f} dB".format(epoch, avg_loss, avg_psnr))

    return avg_loss, avg_psnr


def test():
    avg_psnr = 0
    avg_loss = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)
            prediction = model(input)

            loss = criterion(prediction, target)
            avg_loss += loss.item()
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr

    avg_psnr /= len(testing_data_loader)
    avg_loss /= len(testing_data_loader)
    print("\n===> Avg. Val. Loss: {:.4f} | Avg. Val. PSNR: {:.4f} dB".format(avg_loss, avg_psnr), end="\n\n")

    return avg_loss, avg_psnr


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, opt.nEpochs + 1):
    train_loss, train_psnr = train(epoch)
    test_loss, test_psnr = test()


    if opt.wandb:
        wandb.log({
            "Train Loss": train_loss,
            "Test Loss": test_loss,
            "Train PSNR": train_psnr,
            "Test PSNR": test_psnr,
        })

        if epoch % 10 == 0:
            checkpoint(epoch)
