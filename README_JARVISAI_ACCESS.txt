# JarvisAI Access URLs

- Classic Admin Dashboard: http://localhost:8080/admin/dashboard/admin
- Main SPA Dashboard: http://localhost:8080/

To make the classic admin dashboard the default landing page, run:

    uvicorn main_redirect:app --host 0.0.0.0 --port 8080
