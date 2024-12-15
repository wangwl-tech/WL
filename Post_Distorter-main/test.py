import torch
led_params = torch.Tensor([1, 0.962399, -0.003345, -4348.417859, 0.074754, 0.0055680157]).cuda()
c0 = 1/led_params[0]
c1 = -led_params[1]/led_params[0]
c2 = -led_params[2]/led_params[0]
d0 = 1/led_params[5]
d1 = (led_params[4]**2)/(4*(led_params[5]**2))
d2 = -led_params[4]/(2*led_params[5])
print(c0)
print(c1)
print(c2)
print(d0)
print(d1)
print(d2)
print("*"*40)
print(c0)
print(c1*(-1))
print(c2*290)
print(d0*(1.0/180))
print(d1*(1/46))
print(d2*-0.145)
