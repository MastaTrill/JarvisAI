from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from admin_dashboard import router as admin_dashboard_router

app = FastAPI()

# Mount the admin dashboard router
app.include_router(admin_dashboard_router)


@app.get("/", include_in_schema=False)
def root():
    # Redirect root to the classic admin dashboard
    return RedirectResponse("/admin/dashboard/admin")
