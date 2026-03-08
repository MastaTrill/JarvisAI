def register(api_v1):
    @api_v1.get("/plugin-hello", tags=["Plugins"])
    def plugin_hello():
        return {"message": "Hello from plugin!"}
