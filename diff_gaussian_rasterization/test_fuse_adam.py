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
    def __init__(self, hidden_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

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
    parser.add_argument('--hidden_size', choices=['10', '100', '1000', '10000'], default='1000',
                        help='hidden size of the MLP')
    parser.add_argument('--test_groups', action="store_true", help='Choice of whether test with parameters with multiple groups')
    args = parser.parse_args()

    hidden_size = eval(args.hidden_size)
    x = torch.randn(100, hidden_size).to("cuda:0")
    y = torch.randn(100, 1).to("cuda:0")
    model = SimpleNet(hidden_size).to("cuda:0")
    lrs = {10:1e-1, 100:1e-2, 1000: 1e-4, 10000:1e-5}
    lr = lrs[hidden_size]
    
    params_groups = model.parameters()
    if (args.test_groups):
        params_groups = [{'params': p} for p in model.parameters()]

    if args.optimizer == 'fused_single':
        print("################ Test Fused Adam : Single tensor #################")
        optimizer = FusedAdam(params_groups, lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, weight_decay=0.0, multi_tensor=False)
    elif args.optimizer == 'fused_multi':
        print("################ Test Fused Adam : Multi tensor #################")
        optimizer = FusedAdam(params_groups, lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, weight_decay=0.0, multi_tensor=True)
    elif args.optimizer == 'torch_adam':
        print("################ Test Pytorch Adam #################")
        optimizer = torch.optim.Adam(params_groups, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
    elif args.optimizer == 'torch_adam_fused': 
        print("################ Test Pytorch Adam #################")
        optimizer = torch.optim.Adam(params_groups, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, fused=True)

    print(f"####### Hyper parameters: hidden size: {hidden_size}, lr: {lr}")

    criterion = nn.MSELoss()

    torch.set_printoptions(precision=10)
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
