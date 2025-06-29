# Contributing to Jarvis AI

Thank you for your interest in contributing! This project welcomes code, plugins, documentation, and research contributions.

## How to Contribute

1. **Fork the repository** and create your feature branch.
2. **Write clear, well-documented code** and add tests if possible.
3. **For plugins:**
   - Place your plugin in the `plugins/` folder.
   - Each plugin should define a `register(app)` function to add routes or logic.
   - Example:
     ```python
     def register(app):
         @app.get("/plugin-hello")
         def plugin_hello():
             return {"message": "Hello from plugin!"}
     ```
4. **Run linting and tests locally** before submitting a pull request:
   ```bash
   flake8 .
   pytest
   ```
5. **Submit a pull request** with a clear description of your changes.

## Community Guidelines
- Be respectful and constructive.
- Write clear commit messages.
- Add or update documentation as needed.

## Code of Conduct
See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.
