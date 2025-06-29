class Material:
    """Class representing a material for 3D objects."""

    def __init__(self, name: str, color: tuple, texture: str = None):
        """
        Initialize a Material instance.

        Parameters:
        name (str): The name of the material.
        color (tuple): The RGB color of the material as a tuple (R, G, B).
        texture (str, optional): The file path to the texture image. Defaults to None.
        """
        self.name = name
        self.color = color
        self.texture = texture

    def apply(self):
        """Apply the material properties to the rendering context."""
        # Implementation for applying the material properties
        pass

    def __repr__(self):
        """Return a string representation of the Material."""
        return f"Material(name={self.name}, color={self.color}, texture={self.texture})"