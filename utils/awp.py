import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

EPS = 1E-20

class AdvWeightPerturb:
    def __init__(self, model, delta=0.1, eps=1e-6, use_mixed_precision=False, cfg=None):
        self.model = model
        self.cfg = cfg
        self.delta = delta
        self.eps = eps
        self.use_mixed_precision = use_mixed_precision
        self.scaler = GradScaler() if use_mixed_precision else None

    def calc_awp_and_apply(self, loss_fn, inputs, targets):
        self.model.zero_grad()

        if self.use_mixed_precision:
            with autocast():
                outputs = self.model(inputs)
                loss, ce_loss, dice_loss = loss_fn(outputs, targets)
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(inputs)
            loss, ce_loss, dice_loss = loss_fn(outputs, targets)
            loss.backward()

        # Apply perturbations
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    grad = param.grad
                    perturbation = self.delta * grad / (torch.sqrt(torch.sum(grad ** 2)) + self.eps)
                    param.add_(perturbation)

        return loss, ce_loss, dice_loss

    def restore_weights(self):
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    grad = param.grad
                    perturbation = self.delta * grad / (torch.sqrt(torch.sum(grad ** 2)) + self.eps)
                    param.sub_(perturbation)

    def train_step(self, optimizer, loss_fn, inputs, targets):
        self.calc_awp_and_apply(loss_fn, inputs, targets)

        self.model.zero_grad()
        if self.use_mixed_precision:
            with autocast():
                outputs = self.model(inputs)
                loss, ce_loss, dice_loss = loss_fn(outputs, targets)
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(inputs)
            loss, ce_loss, dice_loss = loss_fn(outputs, targets)
            loss.backward()
        #print("Loss AFTER Perturb: ", loss.item())

        self.restore_weights()

        if self.use_mixed_precision:
            if self.cfg.clip_grad > 0:
                self.scaler.unscale_(optimizer)                          
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
        else:
            if self.cfg.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
            optimizer.step()
            optimizer.zero_grad()


        return loss, ce_loss, dice_loss