import unittest
from src.rendering.engine import RenderingEngine
from src.rendering.shaders import Shader
from src.rendering.materials import Material

class TestRenderingEngine(unittest.TestCase):

    def setUp(self):
        self.engine = RenderingEngine()
        self.shader = Shader(vertex_shader_source="vertex_shader.glsl", fragment_shader_source="fragment_shader.glsl")
        self.material = Material(shader=self.shader)

    def test_initialization(self):
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.current_shader, None)

    def test_add_shader(self):
        self.engine.add_shader(self.shader)
        self.assertEqual(self.engine.current_shader, self.shader)

    def test_apply_material(self):
        self.engine.add_shader(self.shader)
        self.engine.apply_material(self.material)
        self.assertEqual(self.engine.current_material, self.material)

    def test_render_frame(self):
        self.engine.add_shader(self.shader)
        self.engine.apply_material(self.material)
        result = self.engine.render_frame()
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()