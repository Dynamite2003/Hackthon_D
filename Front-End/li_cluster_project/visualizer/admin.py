from django.contrib import admin
from .models import SimulationJob

@admin.register(SimulationJob)
class SimulationJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'status', 'created_at', 'updated_at')
    list_filter = ('status', 'created_at')
    readonly_fields = ('id', 'created_at', 'updated_at')
    search_fields = ('id', 'task_id')