#!/bin/bash

# 启动Celery worker的脚本
# 确保Redis服务器正在运行

echo "🚀 启动LAMMPS项目的Celery worker..."

# 检查Redis是否运行
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis服务器未运行，请先启动Redis:"
    echo "   brew services start redis"
    echo "   或者: redis-server"
    exit 1
fi

echo "✅ Redis服务器正在运行"

# 设置Django环境
export DJANGO_SETTINGS_MODULE=lammps_project.settings

# 启动Celery worker
echo "🔧 启动Celery worker..."
celery -A lammps_project worker --loglevel=info

echo "🛑 Celery worker已停止"