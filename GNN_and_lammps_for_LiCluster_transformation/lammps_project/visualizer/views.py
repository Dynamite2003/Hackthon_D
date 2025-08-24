# visualizer/views.py

from django.http import JsonResponse, HttpResponse, Http404
from django.shortcuts import render # 导入render
from .models import SimulationJob
import os # 导入os

def get_simulation_status(request, job_id):
    """
    一个API端点，用于查询模拟任务的状态。(此函数不变)
    """
    try:
        job = SimulationJob.objects.get(id=job_id)
        return JsonResponse({
            'job_id': job.id,
            'status': job.status,
            'task_id': job.task_id,
            'output_path': job.output_trajectory_path,
            'updated_at': job.updated_at
        })
    except SimulationJob.DoesNotExist:
        return JsonResponse({'error': 'Job not found'}, status=404)


def view_simulation_result(request, job_id):
    """
    一个页面，用于加载并显示最终的可视化结果。
    """
    try:
        job = SimulationJob.objects.get(id=job_id)
    except SimulationJob.DoesNotExist:
        raise Http404("Simulation job not found.")

    context = {
        'job': job,
        'xyz_content': None # 先设为None
    }

    # 只有当任务成功且有输出路径时，才尝试读取文件
    if job.status == SimulationJob.Status.SUCCESS and job.output_trajectory_path:
        # 检查文件是否存在
        if os.path.exists(job.output_trajectory_path):
            with open(job.output_trajectory_path, 'r') as f:
                context['xyz_content'] = f.read()
        else:
            # 如果文件在数据库有记录但实际不存在，也标记为错误
            job.status = SimulationJob.Status.FAILURE
            job.save()

    # 使用render函数渲染一个HTML模板
    return render(request, 'visualizer/result.html', context)