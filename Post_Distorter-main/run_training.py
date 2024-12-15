import torch
import torch.nn as nn
import torch.optim as optim
from data import LED_Distortion
from actor import Led_Actor
from model import Post_distorter
from trainer import Distortion_trainer
from torch.utils.data import DataLoader
led_params = [4.3038e16, 0.9624, -7.7722e-20, -2.3476e-30, 17.3689e-19, 3.0060e-36]
sample_num = 12000
batch_size = 10000
objective = nn.MSELoss()
led_dataset = LED_Distortion('led0', led_params, sample_num)
train_loader = DataLoader(led_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=1, drop_last=True)
net = Post_distorter().cuda()
optimizer = optim.Adam(net.parameters(), lr=1e-4)
led_actor = Led_Actor(net=net, objective=objective)
# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)
led_trainer = Distortion_trainer(actor=led_actor, loader=train_loader, optimizer=optimizer)
led_trainer.train(max_epochs=100000)