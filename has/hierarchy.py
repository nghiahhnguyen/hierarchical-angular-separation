"""Hierarchy routines copied from the paper implementation."""
import json
from pathlib import Path
from typing import Set, Dict, Optional, Tuple, List
import networkx as nx
IMAGENET_HIERARCHY_JSON_PATH = str(Path(__file__).resolve().parents[1] / 'data/imagenet_hierarchy.json')

def load_hierarchy_from_json(
    json_path: str = IMAGENET_HIERARCHY_JSON_PATH,
    prefer_deepest_parent_edges: bool = False,
    deepest_parent_tie_break: str = "json_order",
):
    """
    Load class hierarchy from a JSON file.
    Returns a list of (parent, child) tuples representing edges in the hierarchy graph
    and a dictionary mapping node names to their synset IDs.

    Args:
        json_path: Path to the hierarchy JSON file.
        prefer_deepest_parent_edges: If True, nodes with multiple parents keep only the edge
            that yields the maximal depth for that node (deeper path from a root). When False,
            all parent-child edges from the JSON are returned (default legacy behaviour).
        deepest_parent_tie_break: How to resolve equal-depth multi-parent ties when
            prefer_deepest_parent_edges=True. "json_order" keeps the first parent seen in
            the JSON traversal, "lexicographic" picks the lexicographically smaller parent
            synset ID, and "error" raises.
    """
    import os
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if deepest_parent_tie_break not in {"json_order", "lexicographic", "error"}:
        raise ValueError(
            "deepest_parent_tie_break must be one of "
            "{'json_order', 'lexicographic', 'error'}"
        )

    with open(json_path, 'r') as f:
        hierarchy_data = json.load(f)

    edges: List[Tuple[str, str]] = []
    node_to_synset = {}
    deepest_parent_for_node: Dict[str, Tuple[int, str]] = {}

    # Process the hierarchy data to extract edges and synset IDs
    def extract_edges_and_synsets(node, parent=None, depth: int = 0):
        if 'id' in node:
            node_id = node['id']
            name = node['name']
            node_to_synset[name] = node_id

        # Skip nodes with empty children that are not valid leaf classes
        # Valid leaf classes have an 'index' field; invalid internal nodes don't
        children = node.get('children', [])
        if isinstance(children, list) and len(children) == 0 and parent is not None:
            if 'index' not in node:
                # This is an invalid internal node with no children, skip it
                return

        if parent:
            if prefer_deepest_parent_edges:
                current_best = deepest_parent_for_node.get(node_id)
                if current_best is None or depth > current_best[0]:
                    deepest_parent_for_node[node_id] = (depth, parent)
                elif depth == current_best[0] and parent != current_best[1]:
                    if deepest_parent_tie_break == "lexicographic":
                        if parent < current_best[1]:
                            deepest_parent_for_node[node_id] = (depth, parent)
                    elif deepest_parent_tie_break == "error":
                        raise ValueError(
                            f"Equal-depth multi-parent tie for node {node_id}: "
                            f"{current_best[1]} vs {parent} at depth {depth}"
                        )
            else:
                edges.append((parent, node_id))

        if 'children' in node:
            for child in node['children']:
                extract_edges_and_synsets(child, node_id, depth + 1)

    # Assuming the hierarchy_data is a list of root nodes or a single root object
    if isinstance(hierarchy_data, list):
        for root in hierarchy_data:
            extract_edges_and_synsets(root)
    else:
        extract_edges_and_synsets(hierarchy_data)

    if prefer_deepest_parent_edges:
        edges = [
            (parent, child)
            for child, (depth, parent) in deepest_parent_for_node.items()
            if parent is not None
        ]

    return edges, node_to_synset


def compute_node_depths(G):
    """Compute the depth of each node in the hierarchy from root(s)."""
    if G is None:
        return {}

    # Find root nodes (nodes with no incoming edges)
    root_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]

    if not root_nodes:
        # If no clear roots, use nodes with minimum in-degree
        min_in_degree = min(G.in_degree(node) for node in G.nodes())
        root_nodes = [node for node in G.nodes() if G.in_degree(node) == min_in_degree]

    # Compute depths using BFS from each root
    node_depths = {}
    visited = set()

    for root in root_nodes:
        queue = [(root, 0)]
        while queue:
            node, depth = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            # Keep the maximum depth if node is reachable from multiple paths
            if node in node_depths:
                node_depths[node] = max(node_depths[node], depth)
            else:
                node_depths[node] = depth

            # Add children to queue
            for child in G.successors(node):
                if child not in visited:
                    queue.append((child, depth + 1))

    return node_depths


def build_imagenet_hierarchy_graph(
    hierarchy_path,
    prefer_deepest_parent_edges: bool = False,
    deepest_parent_tie_break: str = "json_order",
):
    """Build NetworkX graph from hierarchy JSON file (supports ImageNet, CIFAR-100, etc.)."""
    print(f"Loading hierarchy from {hierarchy_path}")
    edges, node_to_synset = load_hierarchy_from_json(
        hierarchy_path,
        prefer_deepest_parent_edges=prefer_deepest_parent_edges,
        deepest_parent_tie_break=deepest_parent_tie_break,
    )

    # Build NetworkX graph
    G = nx.DiGraph()
    synset_to_name = {}

    for parent, child in edges:
        G.add_edge(parent, child)
        # Store synset mapping
        for name, synset in node_to_synset.items():
            synset_to_name[synset] = name

    # Remove dataset-specific root nodes that are purely structural/scoping
    # These are root nodes (in-degree 0) that serve as dataset containers with no semantic meaning
    # Examples: 'fall11' for ImageNet, 'cifar100_root' for CIFAR-100
    # Detection heuristics:
    # 1. Node ID matches known scoping patterns (fall11, *_root, etc.)
    # 2. Node is a root with children (not a leaf)
    structural_root_patterns = [
        'fall11',
        'cifar100_root',
        'imagenet_root',
        'tinyimagenet_root',
        'places365_root',
    ]
    structural_roots_to_remove = []
    for node in G.nodes():
        if G.in_degree(node) == 0:  # Root node
            # Check if node ID matches known structural root patterns
            if node in structural_root_patterns or node.endswith('_root'):
                children = list(G.successors(node))
                if len(children) > 0:  # Has children (not a leaf)
                    structural_roots_to_remove.append(node)

    for root in structural_roots_to_remove:
        print(f"Removing structural dataset root node: {root}")
        G.remove_node(root)

    print(f"Hierarchy loaded: {len(G.nodes())} nodes, {len(G.edges())} edges")
    return G, synset_to_name


def collapse_unary_nodes(G: nx.DiGraph, protected_nodes: Optional[Set[str]] = None):
    """
    Collapse internal nodes that have exactly one child by bypassing them,
    connecting their parents directly to their single child, and removing
    the unary node when it becomes isolated.

    Args:
        G: Directed acyclic graph representing the hierarchy. Nodes are synset IDs.
        protected_nodes: Optional set of node IDs that should never be collapsed.

    Returns:
        (collapsed_graph, report_dict)

        report_dict has keys:
            - nodes_before, edges_before, nodes_after, edges_after
            - num_collapsed_nodes
            - collapsed_nodes: list of dicts with details per collapsed node
              { node_id, parents_before, child_before, new_edges_added }
    """
    if G is None:
        return None, {
            "nodes_before": 0,
            "edges_before": 0,
            "nodes_after": 0,
            "edges_after": 0,
            "num_collapsed_nodes": 0,
            "collapsed_nodes": [],
        }

    protected_nodes = protected_nodes or set()

    collapsed_graph = G.copy()

    nodes_before = collapsed_graph.number_of_nodes()
    edges_before = collapsed_graph.number_of_edges()

    collapsed_details = []

    # Iterate until no more collapses are possible
    while True:
        candidates = []
        for node in list(collapsed_graph.nodes()):
            if node in protected_nodes:
                continue
            # Unary node: exactly one child (not a leaf)
            if collapsed_graph.out_degree(node) == 1:
                candidates.append(node)

        if not candidates:
            break

        # Process candidates in this round. Use a snapshot to avoid order effects
        for node in candidates:
            if node not in collapsed_graph:
                continue  # may have been removed in previous operations
            if node in protected_nodes:
                continue

            children = list(collapsed_graph.successors(node))
            if len(children) != 1:
                continue  # not unary anymore
            child = children[0]

            parents = list(collapsed_graph.predecessors(node))
            # If no parents (root), we simply bypass node by connecting nothing to child,
            # but we will remove the node->child edge and delete the node if isolated.
            new_edges_added = []

            for p in parents:
                if p == child:
                    continue  # avoid self-loop
                if not collapsed_graph.has_edge(p, child):
                    collapsed_graph.add_edge(p, child)
                    new_edges_added.append((p, child))
                # Remove p -> node
                if collapsed_graph.has_edge(p, node):
                    collapsed_graph.remove_edge(p, node)

            # Remove node -> child edge
            if collapsed_graph.has_edge(node, child):
                collapsed_graph.remove_edge(node, child)

            # If node is now isolated, remove it
            if collapsed_graph.in_degree(node) == 0 and collapsed_graph.out_degree(node) == 0:
                collapsed_graph.remove_node(node)

            collapsed_details.append({
                "node_id": node,
                "parents_before": parents,
                "child_before": child,
                "new_edges_added": new_edges_added,
            })

    report = {
        "nodes_before": nodes_before,
        "edges_before": edges_before,
        "nodes_after": collapsed_graph.number_of_nodes(),
        "edges_after": collapsed_graph.number_of_edges(),
        "num_collapsed_nodes": len(collapsed_details),
        "collapsed_nodes": collapsed_details,
    }

    return collapsed_graph, report
