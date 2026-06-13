import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(784, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.input(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.log_softmax(self.output(x), dim=1)

device = torch.device("cpu")
model = MNISTnet()
model.load_state_dict(torch.load("model/mnist_model.pth", map_location=device, weights_only=True))
model.eval()

# Dummy input for the ONNX export
dummy_input = torch.randn(1, 1, 28, 28)

torch.onnx.export(
    model, 
    dummy_input, 
    "model/mnist_model.onnx", 
    export_params=True, 
    opset_version=11, 
    do_constant_folding=True, 
    input_names=['input'], 
    output_names=['output'], 
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("Model exported to ONNX.")
