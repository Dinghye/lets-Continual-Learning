"""
A Simple tools to evaluate model's params and FLOPs(MACs) by profile
"""

import torch
from thop.profile import profile
from utils.configs import device

filename = './model/5_increment:3_net.pkl'  # the model that you wanna evaluate
model = torch.load(filename).to(device)
inputs = torch.randn((1, 3, 32, 32)).to(device)
flops, params = profile(model, (inputs,), verbose=False)
print('param:' + str(params) + "|FLOPs:" + str(flops))
