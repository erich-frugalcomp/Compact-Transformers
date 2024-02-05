
import torch
import torch.fx
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

