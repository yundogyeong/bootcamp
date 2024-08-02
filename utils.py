import torch
import torchvision
import torchvision.transforms as transforms
from torch.ao.quantization import QuantStub, DeQuantStub, QuantWrapper
from model import ElementwiseAdd


class QuantAddWrapper(torch.nn.Module):
    def __init__(self, add_module):
        super().__init__()
        self.dequant1 = DeQuantStub()
        self.dequant2 = DeQuantStub()
        self.quant = QuantStub()
    
    def forward(self, a, b):
        x = self.dequant1(a) + self.dequant2(b)
        x = self.quant(x)
        return x
    

def module_wrapper(model):
    model = QuantWrapper(model)
    fuse_list = find_fuse_modules(torch.fx.symbolic_trace(model))
    def helper(parent):
        for name, child in parent.named_children():
            if isinstance(child, ElementwiseAdd):
                setattr(parent, name, QuantAddWrapper(child))
            helper(child)
    helper(model)
    return model, fuse_list


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def find_fuse_modules(graph_module):
    fuse_list = []
    for node in graph_module.graph.nodes:
        main_node = None
        bn_node = None
        relu_node = None
        if node.op == 'call_module' and isinstance(graph_module.get_submodule(node.target), torch.nn.Conv2d) and node.target not in [y for x in fuse_list for y in x]:
            main_node = node
            next_node = main_node.next
            if next_node and next_node.op == 'call_module' and isinstance(graph_module.get_submodule(next_node.target), torch.nn.BatchNorm2d):
                bn_node = next_node
                next_node = bn_node.next
                if next_node and next_node.op == 'call_module' and isinstance(graph_module.get_submodule(next_node.target), torch.nn.ReLU):
                    relu_node = next_node
            elif next_node and next_node.op == 'call_module' and isinstance(graph_module.get_submodule(next_node.target), torch.nn.ReLU):
                relu_node = next_node
        elif node.op == 'call_module' and isinstance(graph_module.get_submodule(node.target), torch.nn.Linear) and node.target not in [y for x in fuse_list for y in x]:
            main_node = node
            next_node = main_node.next
            if next_node and next_node.op == 'call_module' and isinstance(graph_module.get_submodule(next_node.target), torch.nn.ReLU):
                relu_node = next_node
        elif node.op == 'call_module' and isinstance(graph_module.get_submodule(node.target), torch.nn.BatchNorm2d) and node.target not in [y for x in fuse_list for y in x]:
            bn_node = node
            next_node = bn_node.next
            if next_node and next_node.op == 'call_module' and isinstance(graph_module.get_submodule(next_node.target), torch.nn.ReLU):
                relu_node = next_node
        if (fuse_pair := [x.target for x in [main_node, bn_node, relu_node] if x is not None]) != [] and len(fuse_pair) != 1:
            fuse_list.append(fuse_pair)
    return fuse_list


def get_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    return train_loader, test_loader
