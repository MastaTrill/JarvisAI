from fastapi import APIRouter
import subprocess

router = APIRouter(prefix="/automation", tags=["Automation"])

@router.post("/run-all-workflows")
def run_all_workflows():
    result = subprocess.run(["python", "orchestrate_all_workflows.py"], capture_output=True, text=True)
    return {"stdout": result.stdout, "stderr": result.stderr}
