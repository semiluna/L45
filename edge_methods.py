from __future__ import annotations
from typing import Any, Callable, List, Dict

import torch
import torch_cluster

EdgeGeneratorFactory = Callable[[torch.Tensor], torch.Tensor]


def edge_generator(
    coords: torch.Tensor, edge_method: str, edge_method_params: dict[str, Any]
) -> torch.Tensor:
    method = edge_generator_factory(
        edge_method=edge_method, edge_method_params=edge_method_params
    )
    return method(coords)


def edge_generator_factory(
    edge_method: str, edge_method_params: dict[str, Any]
) -> EdgeGeneratorFactory:
    edge_methods = edge_method.split(" + ")
    if len(edge_methods) == 1:
        edge_methods = edge_method.split("+")
        if len(edge_methods) == 1:
            if edge_method in edge_method_params:
                edge_method_params = edge_method_params[edge_method]
            return single_edge_generator_factory(edge_method, edge_method_params)

    # We've got multiple edge methods, so we need to return a function that
    # will combine the returned edges
    edge_gen_funcs = []
    for edge_method in edge_methods:
        single_params = edge_method_params
        if edge_method in edge_method_params:
            single_params = edge_method_params[edge_method]
        edge_gen_funcs.append(single_edge_generator_factory(edge_method, single_params))

    return combine_edge_generator_functions(edge_gen_funcs, edge_method_params)


def single_edge_generator_factory(
    edge_method: str, edge_method_params: dict[str, Any]
) -> EdgeGeneratorFactory:
    def _edge_gen_func(coords: torch.Tensor) -> torch.Tensor:
        if edge_method == "radius":
            default_params = {"r": 10.0, "loop": False}
            params = update_param_dict(default_params, edge_method_params)
            return torch_cluster.radius_graph(coords, **params)
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
        elif edge_method == "random":
            default_params = {
                "num_edges": 40,
                "inverse_temp": 1,
                "probability_distribution": "inverse_cubic",
            }
            params = update_param_dict(default_params, edge_method_params)
            return random_edge_generation(coords, **params)
        else:
            raise ValueError("Unexpected edge method provided: {}".format(edge_method))

    return _edge_gen_func


def combine_edge_generator_functions(
    edge_gen_funcs: List[EdgeGeneratorFactory], edge_method_params: dict[str, Any]
) -> EdgeGeneratorFactory:
    default_params = {"aggregation": "unique"}
    combiner_params = update_param_dict(default_params, edge_method_params)

    def _combiner_func(coords: torch.Tensor) -> torch.Tensor:
        edges = []
        for edge_gen_func in edge_gen_funcs:
            edges.append(edge_gen_func(coords))

        if combiner_params["aggregation"] == "unique":
            concatenated_edge_indices = torch.concatenate(edges, dim=1)
            if concatenated_edge_indices.size(1) <= 1:
                # We have a single edge or no edges
                return concatenated_edge_indices
            temp_idx = torch.argsort(concatenated_edge_indices, dim=-1, stable=True)[-1]
            idx = temp_idx[
                torch.argsort(concatenated_edge_indices[0, temp_idx], stable=True)
            ]
            sorted_cnct_edge_indices = concatenated_edge_indices[:, idx]
            cur_edge_indices = sorted_cnct_edge_indices[:, 1:]
            prev_edge_indices = sorted_cnct_edge_indices[:, :-1]
            unique_edge_indices = sorted_cnct_edge_indices[:, 1:][
                :,
                torch.logical_not(
                    torch.all(cur_edge_indices == prev_edge_indices, dim=0)
                ),
            ]
            return torch.concatenate(
                [sorted_cnct_edge_indices[:, 0:1], unique_edge_indices], dim=1
            )
        elif combiner_params["aggregation"] == "concatenate":
            return torch.concatenate(edges, dim=1)
        else:
            raise ValueError(
                "Unexpected aggregation method for the edges provided: {}".format(
                    combiner_params["aggregation"]
                )
            )

    return _combiner_func


def update_param_dict(
    org_dict: dict[str, Any], new_dict: dict[str, Any]
) -> dict[str, Any]:
    for key in org_dict:
        if key in new_dict:
            # Doing an explicit cast to ensure the type matches
            cast = type(org_dict[key])
            org_dict[key] = cast(new_dict[key])
    return org_dict


def calculate_phi(
    dists: torch.Tensor, probability_distribution: str, **kwargs
) -> torch.Tensor:
    if probability_distribution == "inverse_cubic":
        return -3 * torch.log(dists)
    elif probability_distribution == "uniform":
        return torch.ones_like(dists)
    elif probability_distribution == "exponential":
        default_params = {"scale": 1.0}
        params = update_param_dict(default_params, kwargs)
        scale = params["scale"]
        return -dists * kwargs / scale
    else:
        raise ValueError(
            "Unexpected probability method for random edges provided: {}".format(
                probability_distribution
            )
        )


def random_edge_generation(
    coords: torch.Tensor,
    inverse_temp: float,
    num_edges: int,
    probability_distribution: str,
    **kwargs,
) -> torch.Tensor:
    # coords.shape = [N, 3]
    dists = torch.cdist(coords, coords, p=2)  # [N, N]
    # Setting the diagonal distance values to inf to avoid self-loops here
    dists = dists.fill_diagonal_(torch.inf)
    phi = calculate_phi(dists, probability_distribution, **kwargs)  # [N, N]

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


def convert_to_edge_method_params_dict(
    edge_params: List[str],
) -> Dict[str, Dict[str, Any]]:
    edge_params_dict = dict()
    for param in edge_params:
        method, key, value = param
        if method not in edge_params_dict:
            edge_params_dict[method] = dict()
        edge_params_dict[method][key] = value
    return edge_params_dict
