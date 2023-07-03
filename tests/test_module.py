import pytest
import torch
from torch_compose.module import DirectedModule, ModuleGraph

class DictOutput(DirectedModule):
    def forward(self, x):
        xx = x + 1
        return {'x': xx}
    
class TupleOutput(DirectedModule):
    
    def forward(self, x):
        x_squared = x**2
        return x, x_squared
    
class TensorOutput(DirectedModule):
    
    def forward(self, x):
        return -x
    
class CatModule(DirectedModule):
    
    def forward(self, *args, **kwargs):
        inputs = list(args) + list(kwargs.values())
        return torch.cat(inputs, dim=0)
    
class DictMultiOutput(DirectedModule):
    def forward(self, x):
        x
        return {'x': x, 'y': x+1}



def test_dict_output():
    m = DictOutput(input_keys=['x0'], output_keys = {'x': 'x1'})
    out = m._graph_forward({'x0': torch.tensor([1.0])})
    assert 'x1' in out
    assert torch.equal(out['x1'], torch.tensor([2.0]))

def test_tuple_output():
    m = TupleOutput(input_keys=['x1'], output_keys = ['x2', 'x3'])
    out = m._graph_forward({'x1': torch.tensor([2.0])})
    assert 'x2' in out and 'x3' in out
    assert torch.equal(out['x2'], torch.tensor([2.0]))
    assert torch.equal(out['x3'], torch.tensor([4.0]))

def test_tensor_output():
    m = TensorOutput(input_keys=['x3'], output_keys = ['x4'])
    out = m._graph_forward({'x3': torch.tensor([4.0])})
    assert 'x4' in out
    assert torch.equal(out['x4'], torch.tensor([-4.0]))

def test_module_graph():
    g = ModuleGraph(
        modules={
            # m0 tests a DirectedModule that accepts single string input, performs a simple operation and returns a dict output
            'm0': DictOutput(input_keys=['x0'], output_keys = {'x': 'x1'}),

            # m1 tests a DirectedModule that accepts a dict input and returns a tuple output
            'm1': TupleOutput(input_keys=['x1'], output_keys = ['x2', 'x3']),

            # m2 tests a DirectedModule that accepts a tuple input and returns a Tensor output
            'm2': TensorOutput(input_keys=['x3'], output_keys = ['x4']),

            # cat_layer0 tests a DirectedModule that accepts multiple inputs and concatenates them to return a single output
            'cat_layer0': CatModule(input_keys=['x2', 'x4'], output_keys = ['x5']),

            # m3 tests a DirectedModule that accepts single string input and output and performs a simple operation
            'm3': TensorOutput(input_keys='x5', output_keys = 'x6'),

            # m4 tests a DirectedModule that accepts single string input and returns multiple outputs as a dictionary
            'm4': DictMultiOutput(input_keys='x6', output_keys = {'x': 'x7', 'y': 'x8'}),

            # m5 tests a DirectedModule that accepts a list of input keys and returns a dictionary output
            'm5': DictMultiOutput(input_keys=['x8'], output_keys = {'x': 'x9',}),

            # m6 tests a DirectedModule that accepts a dict input and returns a list output
            'm6': DictMultiOutput(input_keys={'x9': 'x'}, output_keys = ['x']),

            # cat_layer1 tests a DirectedModule that accepts multiple inputs (both single string and list) and concatenates them
            'cat_layer1': CatModule(input_keys=['x7', 'x', 'x8'], output_keys = ['x10']),
            })

    # The forward pass is applied on the input tensor. It should pass through the graph and be transformed according to the defined modules.
    new_batch = g.forward(batch = {'x0': torch.tensor(1.).unsqueeze(0)})

    # Check if the output has the expected key, which indicates that all modules have worked correctly
    assert 'x10' in new_batch

