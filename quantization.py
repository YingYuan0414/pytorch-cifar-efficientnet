import torch
import os

bsz = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load(os.path.join("./experiments/", "499.pth"), map_location=torch.device('cpu'))
print("number of trained parameters: %d" %
          (sum([param.nelement() for param in net.parameters() if param.requires_grad])))
print("number of total parameters: %d" % (sum([param.nelement() for param in net.parameters()])))

model = torch.quantization.quantize_dynamic(net, qconfig_spec=None, dtype=torch.float16, mapping=None, inplace=False)

torch.save(net, os.path.join("./experiments/" + '499_qt1.pth'))