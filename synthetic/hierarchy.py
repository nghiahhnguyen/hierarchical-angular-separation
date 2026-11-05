"""Tree data structure and utilities for hierarchical concept learning."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    """A node in the hierarchy tree."""
    id: int
    depth: int
    parent: Optional['Node'] = None
    children: list['Node'] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None


class Tree:
    """A rooted tree representing a concept hierarchy."""

    def __init__(self, root: Node):
        self.root = root
        self._nodes: dict[int, Node] = {}
        self._index_tree(root)

    def _index_tree(self, node: Node):
        self._nodes[node.id] = node
        for child in node.children:
            self._index_tree(child)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    def get_node(self, node_id: int) -> Node:
        return self._nodes[node_id]

    def get_path_to_root(self, node: Node) -> list[Node]:
        """Return path from node to root (inclusive, node first, root last)."""
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path

    def get_path_ids_to_root(self, node: Node) -> list[int]:
        """Return node IDs on path from node to root."""
        return [n.id for n in self.get_path_to_root(node)]

    def get_siblings(self, node: Node) -> list[Node]:
        """Return sibling nodes (excluding the node itself)."""
        if node.parent is None:
            return []
        return [c for c in node.parent.children if c.id != node.id]

    def get_nodes_at_depth(self, depth: int) -> list[Node]:
        """Return all nodes at given depth."""
        return [n for n in self._nodes.values() if n.depth == depth]

    def get_leaves(self) -> list[Node]:
        """Return all leaf nodes."""
        return [n for n in self._nodes.values() if n.is_leaf()]

    def get_max_depth(self) -> int:
        """Return maximum depth of tree."""
        return max(n.depth for n in self._nodes.values())

    def get_all_nodes(self) -> list[Node]:
        """Return all nodes sorted by ID."""
        return [self._nodes[i] for i in sorted(self._nodes.keys())]


def build_balanced_tree(depth: int, branching_factor: int) -> Tree:
    """
    Build a balanced tree with given depth and branching factor.

    Args:
        depth: Maximum depth (root is depth 0, leaves are at depth `depth`)
        branching_factor: Number of children per internal node

    Returns:
        Tree with nodes numbered 0, 1, 2, ... in BFS order
    """
    node_id = 0
    root = Node(id=node_id, depth=0)
    node_id += 1

    # BFS to build tree level by level
    current_level = [root]
    for d in range(1, depth + 1):
        next_level = []
        for parent in current_level:
            for _ in range(branching_factor):
                child = Node(id=node_id, depth=d, parent=parent)
                parent.children.append(child)
                next_level.append(child)
                node_id += 1
        current_level = next_level

    return Tree(root)
