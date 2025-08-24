from django.db import models
import uuid

class SimulationJob(models.Model):
    class Status(models.TextChoices):
        PENDING = 'PENDING', 'Pending'
        STARTED = 'STARTED', 'Started'
        SUCCESS = 'SUCCESS', 'Success'
        FAILURE = 'FAILURE', 'Failure'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task_id = models.CharField(max_length=255, blank=True, null=True, help_text="Celery task ID")
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PENDING)
    
    # 输入文件路径 (为了演示，我们用CharField，实际项目中可能用FileField)
    structure_a_path = models.CharField(max_length=1024)
    structure_b_path = models.CharField(max_length=1024)
    
    # 输出文件路径
    output_trajectory_path = models.CharField(max_length=1024, blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Job {self.id} - {self.status}"