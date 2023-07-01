from torch import nn
import torch
from typing import Optional, Union, List, Dict
import graphlib
import networkx as nx
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from collections import OrderedDict

class Nodule(nn.Module):
    def __init__(
        self,
        input_keys: Optional[Union[str, List]] = None,
        output_keys: Optional[Union[str, List, Dict]] = None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(input_keys, str):
            input_keys = [input_keys]
        self.input_keys = input_keys
        
        if isinstance(output_keys, str):
            output_keys = [output_keys]
        self.output_keys = output_keys
        
    def _unpack_inputs(self, batch:Dict) -> Dict[str, torch.Tensor]:
        return {k: batch.get(k, None) for k in self.input_keys}
    
    def _update_batch_dict(self, batch:Dict, output:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for k in self.output_keys:
            batch[k] = output[k]
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

class NoduleGraph(nn.ModuleDict):
    
    def __init__(self, modules: Dict[str, Nodule]=None):
        modules = modules or {}
        modules = self._sort_submodules(modules)
        super().__init__(modules = modules)
        

    @staticmethod
    def _sort_submodules(modules):


        graph = {}
        nodule_map = {}
        key_map = {}
        keys = list(modules.keys())

        for ind, nodule in enumerate(modules.values()):

            input_keys = nodule.input_keys
            output_keys = nodule.output_keys


            for key in output_keys:
                graph[key] = set(input_keys)
                key_map[tuple(output_keys)] = keys[ind]
            nodule_map[tuple(output_keys)] = nodule
            
        graph = graph
        sorter = graphlib.TopologicalSorter(graph)
        ordering = [key for key in tuple(sorter.static_order())]


        #

        directed_graph = nx.DiGraph()
        for output, inputs in graph.items():
            for input in inputs:
                directed_graph.add_edge(input, output)
        directed_graph = directed_graph

        ordered_submodules = []
        ordered_keys = []
        for key in ordering:
            for submodule_key in nodule_map.keys():
                if key in submodule_key:
                    ordered_submodules.append(nodule_map.pop(submodule_key))
                    ordered_keys.append(key_map.pop(submodule_key))
                    break

        return OrderedDict({k: v for k, v in zip(ordered_keys, ordered_submodules)})


    def show_graph(self):
        f, ax = plt.subplots(figsize=(12, 10))
        nx.draw(
            self.directed_graph,
            pos=nx.spring_layout(
                self.directed_graph,
            ),
            with_labels=True,
            ax=ax,
        )
