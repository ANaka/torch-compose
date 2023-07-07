# ModuleGraph

## Installation

```bash
pip install torch-compose
```

## Introduction

The ModuleGraph project is a Python-based tool that facilitates managing and manipulating a network of PyTorch modules. The tool employs directed graphs, allowing each module to use the output of another as its input, thus accommodating complex data flows within a neural network.

ModuleGraph is ideal in cases where typical sequential or parallel module arrangements don't suffice, allowing intricate dependencies between modules and handling data propagation in the correct order.

## Features

**DirectedModule**: This versatile abstract base class represents a module that receives and produces data. It requires implementing a forward function within its subclass. It can be integrated into a module graph, providing a controlled data flow from one module to another.

The `DirectedModule` accepts and produces data using defined keys, providing fine control over data partitioning and labeling. It can wrap an existing PyTorch module, reference an existing function as the forward method, or be subclassed for custom forward method implementation, offering great flexibility for various deep learning tasks.


**ModuleGraph**: This class defines a graph of `DirectedModule`s. The data can flow from one module to another according to the topological order of dependencies. The `ModuleGraph` takes care of ensuring that the modules are correctly sorted to respect their dependencies.

The `ModuleGraph` class also provides a visualization method `show_graph()` to visually inspect the graph structure and dependencies using the `networkx` and `matplotlib` libraries. This visual representation shows modules as nodes and dependencies as directed edges, labeled with the corresponding keys.

## Usage

To use the ModuleGraph project, you need to define your `DirectedModule`s and then pass them as a dictionary to the `ModuleGraph` class.

Below is a simplified example:

### Example 1: Using `module` argument to `DirectedModule`

```python
class MyModule(DirectedModule):
    def forward(self, x):
        return torch.relu(x)

# create DirectedModules
mod1 = DirectedModule(input_keys='input', output_keys='hidden', module=MyModule())
mod2 = DirectedModule(input_keys='hidden', output_keys='output', module=MyModule())

# create ModuleGraph
graph = ModuleGraph({'mod1': mod1, 'mod2': mod2})

# forward pass
output = graph.forward({'input': torch.randn(10, 10)})

# visualize the graph
graph.show_graph()
```

`DirectedModule` can be integrated with `torch` code in several additional ways.`

### Example 2: Using `forward` argument to `DirectedModule`

```python
# Defining a function that applies a non-linear activation (ReLU) to its input
def my_relu(input_tensor):
    return torch.nn.functional.relu(input_tensor)

# Create a DirectedModule that wraps this function
relu_module = DirectedModule(input_keys='input', output_keys='output', forward=my_relu)

# Use the module in a graph
graph = ModuleGraph({'relu': relu_module})

# forward pass
output = graph.forward({'input': torch.randn(10, 10)})
```

In this example, the `relu_module` does not encapsulate a PyTorch `nn.Module`, but rather a standalone function that performs an operation using PyTorch functionalities.

### Example 3: Using `DirectedModule` in inheritance

```python
# Defining a custom module by inheriting from both nn.Module and DirectedModule
class MyCustomModule(nn.Module, DirectedModule):
    def __init__(self, input_keys, output_keys):
        nn.Module.__init__(self)
        DirectedModule.__init__(self, input_keys, output_keys)
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

# Create an instance of the custom module
custom_module = MyCustomModule(input_keys='input', output_keys='output')

# Use the module in a graph
graph = ModuleGraph({'custom_module': custom_module})

# forward pass
output = graph.forward({'input': torch.randn(10, 10)})
```

In this example, `MyCustomModule` is a subclass of both `nn.Module` and `DirectedModule`. It uses `nn.Module` to define a simple linear layer and `DirectedModule` to handle input/output key mapping.

### Example 4: Using `DirectedModule` as a mixin

```python
# Defining a custom module by inheriting from both nn.Module and DirectedModule
class MyMixinModule(nn.Module):
    def __init__(self, input_keys, output_keys):
        super().__init__()
        self.directed_module = DirectedModule(input_keys=input_keys, output_keys=output_keys, module=self)
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

# Create an instance of the mixin module
mixin_module = MyMixinModule(input_keys='input', output_keys='output')

# Use the module in a graph
graph = ModuleGraph({'mixin_module': mixin_module.directed_module})

# forward pass
output = graph.forward({'input': torch.randn(10, 10)})
```

In this example, `MyMixinModule` is a subclass of `nn.Module`, and it uses a `DirectedModule` as a member to handle input/output key mapping, allowing the same functionality without multiple inheritance.

