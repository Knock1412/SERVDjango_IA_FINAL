from django.urls import path
from ia_backend.views import summarize_from_url, ask_from_url, get_summarize_status

urlpatterns = [
    path('summarize_from_url/', summarize_from_url),
    path('ask_from_url/', ask_from_url),
    path('get_summarize_status/<str:task_id>/', get_summarize_status),  # ✅ nouveau endpoint async pour Celery

]
 