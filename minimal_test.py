from fastapi import FastAPI
from fastapi.responses import HTMLResponse


app = FastAPI()
api_v1 = APIRouter(prefix="/v1")


@api_v1.get("/", response_class=HTMLResponse)
def root():
    return "<h1>Minimal FastAPI Test - Success</h1>"


app.include_router(api_v1)

if __name__ == "__main__":
    import uvicorn
    import os

    print(f"[DEBUG] CWD: {os.getcwd()}")
    print(f"[DEBUG] __file__: {__file__}")
    uvicorn.run("minimal_test:app", host="0.0.0.0", port=8090, reload=True)
