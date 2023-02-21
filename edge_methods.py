from __future__ import annotations
from typing import Any, Callable

import torch
import torch_cluster

EdgeGeneratorFactory = Callable[[torch.Tensor], torch.Tensor]

def edge_generator_factory(
    edge_method: str, edge_method_params: dict[str, Any]
) -> EdgeGeneratorFactory:
    def _edge_gen_func(coords: torch.Tensor) -> torch.Tensor:
        if edge_method == "radius":
            return torch_cluster.radius_graph(coords, **edge_method_params)
        elif edge_method == "knn":
            default_params = {
                "k": 20,
                "batch": None,
                "loop": False,
                "flow": "source_to_target",
            }
            if "k" not in edge_method_params:
                assert (
                    "n_clusters" in edge_method_params
                ), "Need the number of clusters to perform knn."
                edge_method_params["k"] = edge_method_params["n_clusters"]

            params = update_param_dict(default_params, edge_method_params)

            return torch_cluster.knn_graph(coords, **params)
        elif edge_method == "random_inverse_cubic":
            default_params = {"num_edges": 40, "inverse_temp": 1}
            params = update_param_dict(default_params, edge_method_params)

            return random_inverse_cubic_edge_generation(coords, **params)
        else:
            raise ValueError("Unexpected edge method provided: {}".format(edge_method))

    return _edge_gen_func


def update_param_dict(
    org_dict: dict[str, Any], new_dict: dict[str, Any]
) -> dict[str, Any]:
    for key in org_dict:
        if key in new_dict:
            org_dict[key] = new_dict[key]
    return org_dict


def random_inverse_cubic_edge_generation(
    coords: torch.Tensor, inverse_temp: float, num_edges: int
) -> torch.Tensor:
    # coords.shape = [N, 3]
    dists = torch.cdist(coords, coords, p=2)  # [N, N]
    # Setting the diagonal distance values to inf to avoid self-loops here
    dists = dists.fill_diagonal_(torch.inf)
    phi = -3 * torch.log(dists)  # [N, N]

    epsilon = torch.empty_like(phi).uniform_()
    log_epsilon = torch.log(epsilon)

    zs = inverse_temp * phi - torch.log(-log_epsilon)  # [N, N]

    _, end_indices = torch.topk(
        zs, k=num_edges, dim=1, largest=True, sorted=False
    )  # _, [N, K]
    end_indices = end_indices.reshape((-1))  # [N * K]
    start_indices = torch.arange(coords.size(0))  # [N]
    start_indices = torch.repeat_interleave(start_indices, num_edges)  # [N * K]

    return torch.stack([start_indices, end_indices], dim=0)  # [2, N * K]
