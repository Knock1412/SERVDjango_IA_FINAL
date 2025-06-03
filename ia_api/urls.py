from django.urls import path
from ia_backend.views import summarize_from_url, ask_from_url

urlpatterns = [
    path('summarize_from_url/', summarize_from_url),
    path('ask_from_url/', ask_from_url),
]
