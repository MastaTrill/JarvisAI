# Jarvis Plugin System

This folder is for user-contributed plugins (data processors, models, endpoints).

## How to Add a Plugin
- Place your plugin Python file here.
- Each plugin should define a `register(app)` function to add routes or logic.

## Example Plugin (plugins/example_plugin.py):
```python
def register(app):
    @app.get("/plugin-hello")
    def plugin_hello():
        return {"message": "Hello from plugin!"}
```
