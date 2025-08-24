# lammps_runner/urls.py

from django.urls import path
from . import views

# app_name 用于URL命名空间，可以防止不同app下的URL命名冲突
app_name = 'lammps_runner'

urlpatterns = [
    # 例如，一个用于启动模拟的URL
    # 最终的访问路径会是: /runner/start/
    path('start/', views.start_simulation, name='start_simulation'),
]