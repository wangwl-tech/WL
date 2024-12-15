import torch
import torch.nn

class Basetrainer(object):
  def __init__(self, actor, loader, optimizer, lr_scheduler=None):
    self.epoch = 0
    self.actor = actor
    self.loader = loader
    self.optimizer = optimizer
    if lr_scheduler is not None:
      self.lr_scheduler = lr_scheduler
  def train(self, max_epochs):
    epoch = -1
    for epoch in range(self.epoch + 1, max_epochs + 1):
      print("epoch num: ", epoch)
      self.epoch = epoch
      self.train_epoch()
  def train_epoch(self):
    raise NotImplementedError

class Distortion_trainer(Basetrainer):
  def __init__(self, actor, loader, optimizer, lr_scheduler=None):
    super().__init__(actor, loader, optimizer, lr_scheduler)
  def train_epoch(self):
    flag = False
    for bef_val, cur_val, label in self.loader:
      bef_val = bef_val.cuda()
      cur_val = cur_val.cuda()
      label = label.cuda()
      self.actor.train()
      loss = self.actor(bef_val, cur_val, label)
      # print("loss is : ", loss)
      if flag is False:
        print("loss is : ", loss)
        flag = True
      self.optimizer.zero_grad()
      loss.backward()
      # torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), 1e-3)
      self.optimizer.step()
    return True