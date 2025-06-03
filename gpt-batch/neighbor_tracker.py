from typing import Dict, Literal, Tuple
import numpy as np


class NeighborTracker:
    def __init__(self,
                src: np.ndarray,
                dst: np.ndarray,
                ts: np.ndarray,
                max_size: int = 20,
            ) -> None:
        r"""initialize the neighbor tracker"""
        self.node_dict = {}
        self._check_input(src, dst, ts)
        self.max_size = max_size
        
        #! might not be efficient for large datasets
        for i in range(src.shape[0]):
            if src[i] not in self.node_dict:
                self.node_dict[src[i]] = [(dst[i], ts[i])]
            else:
                self.node_dict[src[i]].append((dst[i], ts[i]))

    def _check_input(self,
                    src: np.ndarray,
                    dst: np.ndarray,
                    ts: np.ndarray,
                    )-> None:
            r"""check if the input is valid"""
            if (src.shape[0] or dst.shape[0] or ts.shape[0]) == 0:
                raise ValueError("Empty input")
            if (src.shape[0] != dst.shape[0] or src.shape[0] != ts.shape[0] or dst.shape[0] != ts.shape[0]):
                raise ValueError("Input shapes do not match")


    def get_neighbor(self, 
                     nodes: np.ndarray,
                     )-> Dict[int, Tuple[int, int]]:
        r"""return the current neighbors for the given nodes"""
        neighbors = {k: self.node_dict[k][-self.max_size:] for k in nodes if k in self.node_dict}
        return neighbors
    
    def update(self,
               src: np.ndarray,
               dst: np.ndarray,
               ts: np.ndarray,
               )-> None:
        r"""update the neighbor tracker with new links"""
        self._check_input(src, dst, ts)
        
        for i in range(src.shape[0]):
            if src[i] not in self.node_dict:
                self.node_dict[src[i]] = [(dst[i], ts[i])]
            else:
                self.node_dict[src[i]].append((dst[i], ts[i]))