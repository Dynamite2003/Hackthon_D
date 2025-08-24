# lammps_runner/views.py

from django.http import JsonResponse
from visualizer.models import SimulationJob
from .tasks import run_tmd_simulation
import os

def start_simulation(request):
    # ================= 关键部分在这里 =================
    # 我们假设项目根目录和manage.py在一起
    # 然后用 os.path.join 构建出输入文件的绝对路径
    base_dir = os.path.abspath(os.path.dirname(__name__)) # 获取项目根目录
    path_a = os.path.join(base_dir, 'media/simulations/inputs/structureA.initial')
    path_b = os.path.join(base_dir, 'media/simulations/inputs/structureB.pdb')
    # =================================================

    if not os.path.exists(path_a) or not os.path.exists(path_b):
        return JsonResponse({'error': 'Input files not found in media/simulations/inputs/.'}, status=404)

    # 1. 创建数据库记录，并将文件路径存入数据库
    new_job = SimulationJob.objects.create(
        structure_a_path=path_a,
        structure_b_path=path_b
    )

    # 2. 发送任务到Celery
    run_tmd_simulation.delay(job_id=str(new_job.id))

    # ... 返回响应 ...
    return JsonResponse({
        'message': 'Simulation task has been submitted.',
        'job_id': new_job.id
    })