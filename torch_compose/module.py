from abc import abstractmethod
from collections import OrderedDict
from graphlib import TopologicalSorter
from typing import Dict, Tuple, Union, Callable

import networkx as nx
import torch
from matplotlib import pyplot as plt
from torch import nn


class DirectedModule(nn.Module):
    """
    An abstract base class for a module that takes a set of inputs and produces a set of outputs.
    The module's forward function needs to be implemented by any subclass. This module can be integrated
    in a module graph where data is flowing from one module to another.
    
    3 ways to play:
    1. Pass in a module argument to wrap a module
    2. Pass in a forward argument to point to a function/method
    3. Subclass or use as a mixin and define a forward method

    Args:
        input_keys: The keys used to extract inputs from the batch.
        output_keys: The keys used to add outputs to the batch.
    """

    def __init__(
        self,
        input_keys: Union[str, Tuple, Dict] = None,  # keys to extract from batch
        output_keys: Union[str, Tuple, Dict] = None,  # keys to add to batch
        module: nn.Module = None,  # module to wrap
        forward: Callable = None,  # function/method to point to
        **kwargs,
    ):
        super().__init__()

        # If input_keys is a single string, convert it to a list for consistency
        if isinstance(input_keys, str):
            input_keys = [input_keys]
        # If input_keys is not a dict, convert it to a tuple to handle multiple inputs
        try:
            input_keys.keys()
        except AttributeError:
            input_keys = tuple(input_keys)
        self.input_keys = input_keys

        # If output_keys is a single string, convert it to a list for consistency
        if isinstance(output_keys, str):
            output_keys = [output_keys]
        # If output_keys is not a dict, convert it to a tuple to handle multiple outputs
        try:
            output_keys.keys()
        except AttributeError:
            output_keys = tuple(output_keys)
        self.output_keys = output_keys
        
        assert module is None or forward is None, "Only one of module or forward can be specified"
        
        if module is not None:
            self.add_module("node", module)
            self.forward = module.forward
        elif forward is not None:
            self.forward = forward
        

    def _graph_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Runs the forward pass for the module, using the specified input and output keys.

        Args:
            batch: The batch of data to be processed.

        Returns:
            The processed batch of data.
        """

        outputs = self._get_forward_outputs(batch)

        # Determine how to process the outputs based on the type of output keys
        if isinstance(self.output_keys, dict):
            self._process_dict_outputs(batch, outputs)
        elif isinstance(self.output_keys, tuple):
            self._process_tuple_outputs(batch, outputs)

        return batch

    def _get_forward_outputs(self, batch):
        """
        Get outputs from the forward method by passing inputs based on their keys.

        Args:
            batch: The batch of data to be processed.

        Returns:
            The outputs of the forward method.
        """

        if isinstance(self.input_keys, Dict):
            # Extract inputs from batch using dict of keys and pass as keyword arguments to the forward method
            inputs = {
                internal_key: batch.get(batch_key, None)
                for batch_key, internal_key in self.input_keys.items()
            }
            return self.forward(**inputs)
        else:  # self.input_keys is a Tuple
            # Extract inputs from batch using tuple of keys and pass as positional arguments to the forward method
            inputs = [batch.get(batch_key, None) for batch_key in self.input_keys]
            return self.forward(*inputs)

    def _process_dict_outputs(self, batch, outputs):
        """
        Processes outputs that are in the form of a dictionary.

        Args:
            batch: The batch of data to be processed.
            outputs: The outputs of the forward method.
        """

        assert isinstance(outputs, Dict)
        for internal_key, batch_key in self.output_keys.items():
            batch[batch_key] = outputs[internal_key]

    def _process_tuple_outputs(self, batch, outputs):
        """
        Processes outputs that are in the form of a tuple.

        Args:
            batch: The batch of data to be processed.
            outputs: The outputs of the forward method.
        """

        if isinstance(outputs, Dict):
            for internal_key, batch_key in zip(self.output_keys, outputs.keys()):
                batch[batch_key] = outputs[internal_key]
        elif isinstance(outputs, Tuple):
            for batch_key, output in zip(self.output_keys, outputs):
                batch[batch_key] = output
        elif isinstance(outputs, torch.Tensor):
            for batch_key in self.output_keys:
                batch[batch_key] = outputs

    @property
    def _input_batch_keys(self):
        """Returns the set of input batch keys"""

        if isinstance(self.input_keys, dict):
            return set(self.input_keys.keys())
        else:
            return set(self.input_keys)

    @property
    def _output_batch_keys(self):
        """Returns the set of output batch keys"""

        if isinstance(self.output_keys, dict):
            return set(self.output_keys.values())
        else:
            return set(self.output_keys)


class ModuleGraph(nn.ModuleDict):
    """
    This class defines a graph of DirectedModules, where each module can take as input the output of another module.

    Args:
        modules: A dictionary of DirectedModule, with keys as the module names and values as the module instances.
    """

    def __init__(self, modules: Dict[str, "DirectedModule"] = None):
        modules = modules or {}
        sorted_modules, graph = self._build_and_sort_graph(modules)
        super().__init__(modules=sorted_modules)
        self.module_graph = graph

    def _build_and_sort_graph(self, modules):
        """
        Builds and sorts the graph of DirectedModules according to their dependencies.

        Args:
            modules: A dictionary of DirectedModule.

        Returns:
            A tuple of sorted modules and the graph of modules.
        """

        graph = self._build_graph(modules)
        sorted_keys = self._sort_keys(graph)
        sorted_modules = self._create_sorted_modules(sorted_keys, modules)
        return sorted_modules, graph

    def _create_sorted_modules(self, sorted_keys, modules):
        """
        Creates an ordered dictionary of modules according to the topological order.

        Args:
            sorted_keys: The keys sorted in topological order.
            modules: A dictionary of DirectedModule.

        Returns:
            An ordered dictionary of modules.
        """

        sorted_modules = OrderedDict()
        for key in sorted_keys:
            if key in modules:
                sorted_modules[key] = modules[key]
        return sorted_modules

    def _build_graph(self, modules):
        """
        Builds the graph of DirectedModules.

        Args:
            modules: A dictionary of DirectedModule.

        Returns:
            The graph of modules.
        """

        graph = nx.DiGraph()
        for name, module in modules.items():
            graph.add_node(name, module=module)
        for name1, module1 in modules.items():
            for name2, module2 in modules.items():
                common_keys = module1._output_batch_keys.intersection(
                    module2._input_batch_keys
                )
                for key in common_keys:
                    graph.add_edge(name1, name2, key=key)

        # Sort nodes by topological order
        sorted_nodes = self._sort_keys(graph)

        # Add a final dummy node
        final_node_name = "output"
        graph.add_node(final_node_name)
        last_node_name = sorted_nodes[-1]  # Get the name of the last node

        # Add edges from the last node to the final dummy node for all output keys of the last node
        last_module = modules[last_node_name]
        for key in last_module._output_batch_keys:
            graph.add_edge(last_node_name, final_node_name, key=key)
        return graph

    @staticmethod
    def _sort_keys(graph):
        """
        Sorts the keys in the graph in topological order.

        Args:
            graph: The graph of modules.

        Returns:
            A list of keys in topological order.
        """

        graph_dict = {node: set(graph.predecessors(node)) for node in graph.nodes()}
        return list(TopologicalSorter(graph_dict).static_order())

    def forward(self, batch: dict):
        """
        Runs the forward pass for all modules in the graph in topological order.

        Args:
            batch: The batch of data to be processed.

        Returns:
            The processed batch of data.
        """

        for module_name, module in self.items():
            batch = module._graph_forward(batch)
        return batch

    def show_graph(self):
        """
        Shows the graph of modules using networkx and matplotlib.
        """

        f, ax = plt.subplots(figsize=(12, 10))

        # use a spring layout to spread the nodes out
        pos = nx.planar_layout(self.module_graph)

        # draw the nodes as small black dots and the labels in larger text
        nx.draw_networkx_nodes(
            self.module_graph, pos, node_size=50, node_color="blue", ax=ax
        )

        # draw the edges as thicker lines with arrows at the end
        nx.draw_networkx_edges(
            self.module_graph,
            pos,
            node_size=50,
            arrowstyle="->",
            arrowsize=10,
            edge_cmap=plt.cm.Blues,
            width=2,
            ax=ax,
        )

        # offset the label position to avoid overlap with nodes
        label_pos = {
            k: (v[0] + 0.1, v[1] + 0.05) for k, v in pos.items()
        }  # Adjust second value to offset labels

        # draw labels in larger text
        nx.draw_networkx_labels(
            self.module_graph, label_pos, font_size=20, ax=ax, font_color="blue"
        )

        # draw edge labels
        edge_labels = {
            (u, v): data["key"] for u, v, data in self.module_graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            self.module_graph, pos, edge_labels=edge_labels, ax=ax, font_size=20
        )

        plt.show()
