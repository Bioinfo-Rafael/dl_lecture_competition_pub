
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class EEGNet(nn.Module):
        def __init__(self, num_classes, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
            super(EEGNet, self).__init__()
            self.num_classes = num_classes

            # First Conv2D Layer
            self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=0, bias=False)
            self.batchnorm1 = nn.BatchNorm2d(F1, False)
            
            # Depthwise Conv2D Layer
            self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
            self.batchnorm2 = nn.BatchNorm2d(F1 * D, False)
            
            # Average Pooling Layer
            self.avgpool1 = nn.AvgPool2d((1, 4))
            self.dropout1 = nn.Dropout(dropoutRate)
            
            # Separable Conv2D Layer
            self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
            self.batchnorm3 = nn.BatchNorm2d(F2, False)
            
            # Average Pooling Layer
            self.avgpool2 = nn.AvgPool2d((1, 8))
            self.dropout2 = nn.Dropout(dropoutRate)

            # Fully Connected Layer
            self.fc1 = nn.Linear(F2 * ((Samples // 32) - 1), num_classes)

        def forward(self, x):
            x = x.unsqueeze(1)  # Add a channel dimension
            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = F.elu(x)
            x = self.depthwiseConv(x)
            x = self.batchnorm2(x)
            x = F.elu(x)
            x = self.avgpool1(x)
            x = self.dropout1(x)
            x = self.separableConv(x)
            x = self.batchnorm3(x)
            x = F.elu(x)
            x = self.avgpool2(x)
            x = self.dropout2(x)
            x = x.flatten(start_dim=1)
            x = self.fc1(x)
            return x

    # Usage example
    # Assuming train_set.num_classes = 2, train_set.seq_len = 128, train_set.num_channels = 64
    model = EEGNet(
        num_classes=train_set.num_classes,
        Chans=train_set.num_channels,
        Samples=train_set.seq_len
    ).to(args.device)


    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
            
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

if __name__ == "__main__":
    run()