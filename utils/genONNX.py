
import torch
from utils import saverloader
from model.echopips import EchoPIPs



echotracker = EchoPIPs()
_ = saverloader.load("model/weights/", echotracker)

rgbs = torch.randn(1, 124, 1, 256, 256) # B, S, C, H, W
query_points = torch.randn(1, 85, 2) # B, N, 2 -> (x, y) in pixels. From the first frame.


#torch.onnx.export(echotracker, (rgbs, query_points), "echotracker.onnx", export_params=True, opset_version=11)
export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
onnx_program = torch.onnx.dynamo_export(echotracker, rgbs, query_points, export_options=export_options)
onnx_program.save("echotracker.onnx")


# import torch
# import torch.nn as nn

# class MLPModel(nn.Module):
#   def __init__(self):
#       super().__init__()
#       self.fc0 = nn.Linear(8, 8, bias=True)
#       self.fc1 = nn.Linear(8, 4, bias=True)
#       self.fc2 = nn.Linear(4, 2, bias=True)
#       self.fc3 = nn.Linear(2, 2, bias=True)

#   def forward(self, tensor_x: torch.Tensor):
#       tensor_x = self.fc0(tensor_x)
#       tensor_x = torch.sigmoid(tensor_x)
#       tensor_x = self.fc1(tensor_x)
#       tensor_x = torch.sigmoid(tensor_x)
#       tensor_x = self.fc2(tensor_x)
#       tensor_x = torch.sigmoid(tensor_x)
#       output = self.fc3(tensor_x)
#       return output

# model = MLPModel()
# tensor_x = torch.rand((97, 8), dtype=torch.float32)
# onnx_program = torch.onnx.dynamo_export(model, tensor_x)
# onnx_program.save("mlp.onnx")