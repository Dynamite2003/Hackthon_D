# visualizer/urls.py

from django.urls import path
from . import views

app_name = 'visualizer'

urlpatterns = [
    path('status/<uuid:job_id>/', views.get_simulation_status, name='simulation_status'),
    path('result/<uuid:job_id>/', views.view_simulation_result, name='simulation_result'),
]