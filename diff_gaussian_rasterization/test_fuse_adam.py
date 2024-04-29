import torch
import time
import torch.nn as nn
from diff_gaussian_rasterization import FusedAdam

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10000, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet().to("cuda:0")
# print("################ Test Fused Adam #################")
# optimizer = FusedAdam(model.parameters(), lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, weight_decay=0.0)
print("################ Test Pytorch Adam #################")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)


criterion = nn.MSELoss()

torch.set_printoptions(precision=10)
x = torch.randn(100, 10000).to("cuda:0")
y = torch.randn(100, 1).to("cuda:0")

total_time = 0
epochs = 20
for epoch in range(20):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    t1 = time.time()
    optimizer.step()
    t2 = time.time()
    total_time += (t2 - t1)

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
print(f"Adam optimizer AVG time: {total_time/epochs}")