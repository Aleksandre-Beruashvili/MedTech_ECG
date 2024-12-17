# ecg_analysis/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_ecg, name='upload'),  # Upload view
    path('<int:ecg_id>/', views.diagnosis, name='diagnosis'),  # Diagnosis view
]
