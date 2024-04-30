import torch
import time
import torch.nn as nn
from diff_gaussian_rasterization import FusedAdam
import torch.cuda.profiler as profiler
import argparse

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1000, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

def main():
    parser = argparse.ArgumentParser(description='Select an optimizer.')
    parser.add_argument('--optimizer', choices=['fused_single', 'fused_multi', 'torch_adam', 'torch_adam_fused'], required=True,
                        help='Select the optimizer: "fused_single" for FusedAdam single tensor, "fused_multi" for FusedAdam multi tensor, "torch_adam" for PyTorch Adam, or "torch_adam_fused" for PyTorch FusedAdam.')
    args = parser.parse_args()

    model = SimpleNet().to("cuda:0")
    
    if args.optimizer == 'fused_single':
        print("################ Test Fused Adam : Single tensor #################")
        optimizer = FusedAdam(model.parameters(), lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, weight_decay=0.0, multi_tensor=False)
    elif args.optimizer == 'fused_multi':
        print("################ Test Fused Adam : Multi tensor #################")
        optimizer = FusedAdam(model.parameters(), lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, weight_decay=0.0, multi_tensor=True)
    elif args.optimizer == 'torch_adam':
        print("################ Test Pytorch Adam #################")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
    elif args.optimizer == 'torch_adam_fused': 
        print("################ Test Pytorch Adam #################")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, fused=True)
    criterion = nn.MSELoss()

    torch.set_printoptions(precision=10)
    x = torch.randn(100, 1000).to("cuda:0")
    y = torch.randn(100, 1).to("cuda:0")

    total_time = 0
    epochs = 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        t1 = time.time()
        profiler.start()
        optimizer.step()
        profiler.stop()
        t2 = time.time()
        total_time += (t2 - t1)
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    print(f"Adam optimizer AVG time: {(total_time/epochs)*1000} ms")

if __name__ == "__main__":
    main()
