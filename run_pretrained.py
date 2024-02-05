
import torch
import torch.fx
import torchvision.io
import torchvision.transforms

from src import cct_14_7x2_224

model = cct_14_7x2_224(pretrained=True, progress=True)

model = model.eval()

m2 = torch.fx.symbolic_trace(model)

class InliningTracer(torch.fx.Tracer):
    def __init__(self):
        super().__init__()
    def is_leaf_module(self,m,qname): return False


def inline(m):
    my_tracer = InliningTracer()
    newgraph = my_tracer.trace(m)
    name = m.__class__.__name__
    asmod = torch.fx.GraphModule(my_tracer.root,newgraph,class_name=name)
    asmod.recompile()
    return asmod

m3 = inline(m2)

m3.graph.print_tabular()

imtrans = torchvision.models.get_model_weights('ResNet18').DEFAULT.transforms()
#im = torchvision.io.read_image('cropped_photo.png',torchvision.io.ImageReadMode.RGB).unsqueeze(0).float()
#mymean = torch.mean(im)
#mystd = torch.std(im)
#im = torchvision.transforms.Normalize(mymean,mystd)(im)

#im = torchvision.io.read_image('cropped_photo.png',torchvision.io.ImageReadMode.RGB).unsqueeze(0)
im = torchvision.io.read_image('example.jpg',torchvision.io.ImageReadMode.RGB).unsqueeze(0)
im = imtrans(im)

print(im.size())

ret = model(im).flatten()

top1_frugal_idx = ret.argmax()
top1_frugal_val = ret[top1_frugal_idx]

print(f"*****   REF  IMPL:      Top: {top1_frugal_idx} val={top1_frugal_val}")


