class Lighting:
    """Class to handle light sources and their properties in the rendering system."""

    def __init__(self):
        """Initialize the Lighting class with default light properties."""
        self.lights = []

    def add_light(self, light):
        """Add a light source to the lighting system.

        Args:
            light: An object representing a light source.
        """
        self.lights.append(light)

    def remove_light(self, light):
        """Remove a light source from the lighting system.

        Args:
            light: An object representing a light source to be removed.
        """
        self.lights.remove(light)

    def update_lights(self):
        """Update the properties of all light sources."""
        for light in self.lights:
            light.update()

    def get_light_data(self):
        """Retrieve data for all light sources.

        Returns:
            A list of light properties.
        """
        return [light.get_properties() for light in self.lights]