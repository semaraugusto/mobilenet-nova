# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

from torcheval.metrics.functional import multiclass_accuracy


class SeparableConv2d(nn.Module):
    '''Separable convolution'''
    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConv2d, self).__init__()
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
        )
        self.pw_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x

class MyMobileNet(pl.LightningModule):
    cfg = [
        (32, 64, 1), 
        (64, 128, 2), 
        (128, 128, 1), 
        (128, 256, 2),
        (256, 256, 1),
        (256, 512, 2),
        (512, 512, 1),
        (512, 512, 1),
        (512, 512, 1),
        (512, 512, 1),
        (512, 512, 1),
        (512, 1024, 2),
        (1024, 1024, 1),
    ]
    
    def __init__(self, steps_per_epoch, num_classes: int=10, alpha: float=1, max_epochs: int=50):
        super(MyMobileNet, self).__init__()
        conv_out = int(32 * alpha)
        self.conv = nn.Conv2d(3, conv_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(conv_out)
        self.relu = nn.ReLU(inplace=False)
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.accuracy = multiclass_accuracy

        self.features = self.make_feature_extractor(alpha)
        self.linear = nn.Linear(int(1024*alpha), num_classes)

    def make_feature_extractor(self, alpha):
        layer_values = [(int(inp*alpha), int(out*alpha), chan) for inp, out, chan in self.cfg]
        layers = nn.Sequential(*[SeparableConv2d(*tup) for tup in layer_values])
        return layers

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.features(x)
        x = F.avg_pool2d(x, 2)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x
        
    def step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.compute_loss(logits, y)
        acc = self.accuracy(logits, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        loss, acc = self.step(batch, batch_idx)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def configure_optimizers(self):
        # Torch find_lr suggestion
        self.lr = 0.02089
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.3, total_steps=self.max_epochs * self.steps_per_epoch)
        return { "optimizer": optimizer, "lr_scheduler": scheduler }
        
    def compute_loss(self, logits, labels):
        return nn.functional.cross_entropy(logits, labels)
