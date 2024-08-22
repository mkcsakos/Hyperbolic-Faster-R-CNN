import time
import torch

t = 0

max = 7
min = 3

print("Training In Progress.....")

while True:
    tensor = (max-min)*torch.rand((5, 3)) + min
    tensor.to('cuda')

    t += 1

    print("checkpoint - ", t, tensor)

    time.sleep(120)
