from torch import nn
import torch
from typing import Optional, Union, List, Dict
import networkx as nx
from matplotlib import pyplot as plt
from graphlib import TopologicalSorter
from abc import ABC, abstractmethod
from collections import OrderedDict

class DirectedModule(nn.Module):
    def __init__(
        self,
        input_keys: Union[str, List] = None, # keys to extract from batch
        output_keys: Union[str, List, Dict] = None, # keys to add to batch
        input_key_remapping: Optional[Dict[str, str]] = None, # remap input keys to kwargs for _forward; should not be needed in most cases
        output_key_remapping: Optional[Dict[str, str]] = None, # remap output keys to kwargs for _forward; should not be needed in most cases
        **kwargs,
    ):
        super().__init__()
        if isinstance(input_keys, str):
            input_keys = [input_keys]
        self.input_keys = input_keys
        
        if isinstance(output_keys, str):
            output_keys = [output_keys]
        self.output_keys = output_keys
        
        # by default, input_key_remapping and output_key_remapping are identity mappings
        input_key_remapping = input_key_remapping or {}
        for batch_key in self.input_keys:
            if batch_key not in input_key_remapping.keys():
                input_key_remapping[batch_key] = batch_key
        self.input_key_remapping = input_key_remapping
            
        output_key_remapping = output_key_remapping or {}
        for batch_key in self.output_keys:
            if batch_key not in output_key_remapping.keys():
                output_key_remapping[batch_key] = batch_key
        self.output_key_remapping = output_key_remapping
        
    def _unpack_inputs(self, batch:Dict) -> Dict[str, torch.Tensor]:
        return {internal_key: batch.get(batch_key, None) for batch_key, internal_key in self.input_key_remapping.items()}
    
    def _update_batch_dict(self, batch:Dict, output:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for batch_key, internal_key in self.output_key_remapping.items():
            batch[batch_key] = output[internal_key]
        return batch

    def forward(self, batch:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        outputs = None
        inputs = self._unpack_inputs(batch)
        outputs = self._forward(**inputs)
        batch = self._update_batch_dict(batch, outputs)
        
        return batch
    
    @abstractmethod
    def _forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        pass


class ModuleGraph(nn.ModuleDict):
    def __init__(self, modules: Dict[str, 'DirectedModule']=None):
        modules = modules or {}
        modules, directed_graph = self._sort_submodules(modules)
        super().__init__(modules=modules)
        self.directed_graph = directed_graph
        
    @staticmethod
    def _build_graph(modules):
        graph = nx.DiGraph()
        for name, module in modules.items():
            for output_key in module.output_keys:
                for input_key in module.input_keys:
                    graph.add_edge(name + '.' + output_key, name + '.' + input_key, key=input_key)
        return graph

    @staticmethod
    def _sort_keys(graph):
        graph_dict = {node: set(graph.predecessors(node)) for node in graph.nodes()}
    
        return list(TopologicalSorter(graph).static_order())

    @staticmethod
    def _create_directed_graph(graph):
        directed_graph = nx.DiGraph()
        for output, inputs in graph.items():
            for input in inputs:
                directed_graph.add_edge(input, output)
        return directed_graph


    def _sort_submodules(self, modules):
        graph = self._build_graph(modules)
        sorted_keys = self._sort_keys(graph)
        
        sorted_modules = OrderedDict()
        for key in sorted_keys:
            module_name = key.split('.')[0]  # Assumes module names don't contain '.'
            if module_name in modules:
                sorted_modules[module_name] = modules[module_name]
        
        return sorted_modules, graph
    
    def forward(self, batch: dict):
        for module_name, module in self.items():
            batch = module(batch)
        return batch

    def show_graph(self):
        f, ax = plt.subplots(figsize=(12, 10))
        edge_labels = nx.get_edge_attributes(self.module_graph, 'key')
        pos = nx.spring_layout(self.module_graph)
        nx.draw(self.module_graph, pos, with_labels=True, ax=ax)
        nx.draw_networkx_edge_labels(self.module_graph, pos, edge_labels=edge_labels, ax=ax)
