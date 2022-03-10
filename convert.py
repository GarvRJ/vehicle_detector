import torch
import torchvision.models as models

model = models.(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 244, 244)

input_names = [""]
output_names = [""]

torch.onnx.export(model,
                  dummy_input,
                  "resnet50.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  )