class SceneGraph:
    """Manages the hierarchy of objects in the scene."""

    def __init__(self):
        """Initializes an empty scene graph."""
        self.nodes = []

    def add_node(self, node):
        """Adds a node to the scene graph.

        Args:
            node: The node to be added to the scene graph.
        """
        self.nodes.append(node)

    def remove_node(self, node):
        """Removes a node from the scene graph.

        Args:
            node: The node to be removed from the scene graph.
        """
        self.nodes.remove(node)

    def traverse(self):
        """Traverses the scene graph and performs an action on each node."""
        for node in self.nodes:
            node.update()  # Assuming each node has an update method

    def get_nodes(self):
        """Returns the list of nodes in the scene graph.

        Returns:
            List of nodes in the scene graph.
        """
        return self.nodes

    def clear(self):
        """Clears all nodes from the scene graph."""
        self.nodes.clear()