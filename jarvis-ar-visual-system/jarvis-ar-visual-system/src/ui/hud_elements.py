from dataclasses import dataclass

@dataclass
class HealthBar:
    max_health: int
    current_health: int

    def draw(self, x: int, y: int) -> None:
        """Draws the health bar on the screen at the specified position."""
        # Implementation for rendering the health bar
        pass

    def update(self, health: int) -> None:
        """Updates the current health of the health bar."""
        self.current_health = max(0, min(health, self.max_health))

@dataclass
class ScoreDisplay:
    score: int

    def draw(self, x: int, y: int) -> None:
        """Draws the score display on the screen at the specified position."""
        # Implementation for rendering the score display
        pass

    def update(self, score: int) -> None:
        """Updates the score display with the new score."""
        self.score = score

# Additional HUD elements can be defined here as needed.