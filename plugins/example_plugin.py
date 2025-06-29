def register(app):
    @app.get("/plugin-hello", tags=["Plugins"])
    def plugin_hello():
        return {"message": "Hello from plugin!"}
