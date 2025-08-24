django项目名为lammps_project

三个应用为GNN(尚未创建） lammps_runner visualizer

GNN产生的文件xx.initial xx.pdb在/media/simulations/<job—id>/inputs下

lammps_runner生成的in.tmd trajectory.xyz在/media/simulations/outputs

visualizer负责可视化

网站代码在/visualizer/templates/visulizer/中，具体连接各个应用的views.py使用

依赖项:

sudo apt-get install redis-server

启动并检查检查Redis服务状态

sudo service redis-server start

redis-cli ping

如果返回 PONG，则表示Redis正在运行

python环境django celery redis flower
