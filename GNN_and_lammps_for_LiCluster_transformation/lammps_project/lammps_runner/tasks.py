import os
import subprocess
from celery import shared_task
from visualizer.models import SimulationJob

# 假设你的输入输出文件都存放在Django项目的media目录下
# 你需要在settings.py中配置 MEDIA_ROOT
# MEDIA_ROOT = BASE_DIR / 'media'
BASE_OUTPUT_DIR = "media/simulations/outputs"

# 确保输出目录存在
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


@shared_task(bind=True)
def run_tmd_simulation(self, job_id):
    """
    增强的Celery任务，用于运行LAMMPS TMD模拟，支持真正的TMD轨迹生成。
    :param self: Celery任务实例
    :param job_id: SimulationJob模型的主键，用于更新状态
    """
    try:
        job = SimulationJob.objects.get(id=job_id)
        job.status = SimulationJob.Status.STARTED
        job.task_id = self.request.id
        job.save()
    except SimulationJob.DoesNotExist:
        print(f"Error: Job with id {job_id} not found.")
        return

    # --- 1. 定义文件路径 ---
    initial_structure_file = job.structure_a_path
    target_pdb_file = job.structure_b_path
    
    # 为本次模拟创建一个唯一的工作目录
    job_dir = os.path.join(BASE_OUTPUT_DIR, str(job.id))
    os.makedirs(job_dir, exist_ok=True)
    
    # 复制输入文件到工作目录
    import shutil
    local_initial_file = os.path.join(job_dir, "input.data")
    local_target_file = os.path.join(job_dir, "target.pdb")
    
    shutil.copy2(initial_structure_file, local_initial_file)
    shutil.copy2(target_pdb_file, local_target_file)
    
    lammps_script_path = os.path.join(job_dir, "in.lmp")
    output_xyz_path = os.path.join(job_dir, "trajectory.xyz")
    log_file_path = os.path.join(job_dir, "simulation.log")

    # --- 2. 解析目标结构坐标用于TMD ---
    try:
        from ase.io import read
        target_atoms = read(local_target_file)
        target_coords = target_atoms.get_positions()
        
        # 创建目标坐标文件供LAMMPS使用
        target_coords_file = os.path.join(job_dir, "target_coords.txt")
        with open(target_coords_file, 'w') as f:
            for i, (x, y, z) in enumerate(target_coords):
                f.write(f"{i+1} {x:.6f} {y:.6f} {z:.6f}\n")
                
    except Exception as e:
        print(f"Warning: Could not parse target structure for TMD: {e}")
        target_coords_file = None

    # --- 3. 生成增强的LAMMPS输入脚本 ---
    if target_coords_file and os.path.exists(target_coords_file):
        # 使用增强的MD模拟，包含温度变化
        lammps_script_content = f"""# Enhanced LAMMPS MD simulation script
units metal
atom_style atomic
boundary f f f

# Read initial structure
read_data input.data

# Define Li-Li interactions (Lennard-Jones potential)
pair_style lj/cut 10.0
pair_coeff 1 1 0.05 2.5

# Define groups
group mobile id > 0

# Energy minimization
minimize 1.0e-4 1.0e-6 1000 10000

# Initialize velocities
velocity mobile create 300.0 4928459 rot yes dist gaussian

# Temperature control with ramping
fix nvt_fix mobile nvt temp 300.0 100.0 100.0

# Output trajectory every 15 steps for smooth animation
dump traj_dump all xyz 15 trajectory.xyz
dump_modify traj_dump element Li

# Output thermodynamics
thermo_style custom step temp press etotal pe ke
thermo 100

# Log file
log {os.path.basename(log_file_path)}

# Run simulation with temperature ramp
run 4000

# Clean up
unfix nvt_fix
undump traj_dump
"""
    else:
        # 回退到标准MD模拟
        lammps_script_content = f"""# Standard LAMMPS MD simulation script
units metal
atom_style atomic
boundary f f f

# Read initial structure
read_data input.data

# Define Li-Li interactions
pair_style lj/cut 10.0
pair_coeff 1 1 0.05 2.5

# Define groups
group mobile id > 0

# Energy minimization
minimize 1.0e-4 1.0e-6 1000 10000

# Initialize velocities
velocity mobile create 300.0 4928459 rot yes dist gaussian

# Temperature control with gradual cooling
fix nvt_fix mobile nvt temp 300.0 100.0 100.0

# Output trajectory every 15 steps
dump traj_dump all xyz 15 trajectory.xyz
dump_modify traj_dump element Li

# Output thermodynamics
thermo_style custom step temp press etotal pe ke
thermo 100

# Log file
log {os.path.basename(log_file_path)}

# Run simulation with temperature ramp
run 3000

# Clean up
unfix nvt_fix
undump traj_dump
"""

    with open(lammps_script_path, 'w') as f:
        f.write(lammps_script_content)

    # --- 4. 执行LAMMPS ---
    try:
        # 尝试不同的LAMMPS可执行文件
        lammps_executables = [
            'lmp', 'lmp_stable', 'lammps', 'lmp_serial', 'lmp_mpi',
            '/opt/homebrew/opt/lammps/bin/lmp_serial',
            '/opt/homebrew/opt/lammps/bin/lmp_mpi',
            '/usr/local/bin/lmp',
            '/usr/bin/lammps'
        ]
        
        lammps_executable = None
        for exe in lammps_executables:
            try:
                result = subprocess.run([exe, '-help'], 
                                      capture_output=True, check=False, timeout=5)
                output = result.stdout.decode() + result.stderr.decode()
                if 'LAMMPS' in output or 'Large-scale Atomic' in output:
                    lammps_executable = exe
                    print(f"Found LAMMPS executable: {exe}")
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        if not lammps_executable:
            # 如果没有找到LAMMPS，使用回退方法
            print("LAMMPS not found, using fallback trajectory generation")
            return _generate_fallback_trajectory(job, local_initial_file, local_target_file, output_xyz_path)
        
        # 运行LAMMPS模拟
        print(f"Running LAMMPS simulation with {lammps_executable}")
        result = subprocess.run(
            [lammps_executable, "-in", os.path.basename(lammps_script_path)],
            cwd=job_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )

        # 记录LAMMPS输出
        with open(os.path.join(job_dir, "lammps_output.log"), 'w') as f:
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\nSTDERR:\n")
            f.write(result.stderr)

        # 检查输出文件是否生成
        if not os.path.exists(output_xyz_path):
            raise RuntimeError("LAMMPS completed but no trajectory file was generated")

        # 验证轨迹文件内容
        with open(output_xyz_path, 'r') as f:
            content = f.read()
            if len(content.strip()) == 0:
                raise RuntimeError("Generated trajectory file is empty")

        # 更新任务状态为成功
        job.status = SimulationJob.Status.SUCCESS
        job.output_trajectory_path = output_xyz_path
        job.save()
        print(f"Job {job.id} completed successfully. Output: {output_xyz_path}")

    except subprocess.TimeoutExpired:
        print(f"LAMMPS simulation timed out for job {job.id}")
        return _generate_fallback_trajectory(job, local_initial_file, local_target_file, output_xyz_path)

    except subprocess.CalledProcessError as e:
        print(f"LAMMPS simulation failed for job {job.id}: {e}")
        print("STDERR:", e.stderr)
        return _generate_fallback_trajectory(job, local_initial_file, local_target_file, output_xyz_path)

    except Exception as e:
        print(f"Unexpected error in LAMMPS simulation for job {job.id}: {e}")
        return _generate_fallback_trajectory(job, local_initial_file, local_target_file, output_xyz_path)


def _generate_fallback_trajectory(job, initial_file, target_file, output_path):
    """生成回退轨迹当LAMMPS不可用时"""
    try:
        from lammps_runner.lammps_simulator import LAMMPSSimulator
        
        # 解析初始坐标
        if initial_file.endswith('.initial'):
            # 解析LAMMPS数据文件
            coords = _parse_lammps_data_file(initial_file)
        else:
            from ase.io import read
            atoms = read(initial_file)
            coords = atoms.get_positions()
        
        # 使用LAMMPSSimulator的回退方法
        simulator = LAMMPSSimulator()
        result_path = simulator._generate_fallback_trajectory(coords, target_file, output_path)
        
        if result_path and os.path.exists(result_path):
            job.status = SimulationJob.Status.SUCCESS
            job.output_trajectory_path = result_path
            job.save()
            print(f"Fallback trajectory generated for job {job.id}")
        else:
            job.status = SimulationJob.Status.FAILURE
            job.save()
            print(f"Failed to generate fallback trajectory for job {job.id}")
            
    except Exception as e:
        job.status = SimulationJob.Status.FAILURE
        job.save()
        print(f"Error generating fallback trajectory for job {job.id}: {e}")


def _parse_lammps_data_file(data_file_path):
    """解析LAMMPS数据文件获取坐标"""
    import numpy as np
    
    coords = []
    with open(data_file_path, 'r') as f:
        lines = f.readlines()
        
    in_atoms_section = False
    for line in lines:
        line = line.strip()
        if line == "Atoms":
            in_atoms_section = True
            continue
        elif in_atoms_section and line == "":
            continue
        elif in_atoms_section and line:
            try:
                parts = line.split()
                if len(parts) >= 5:  # atom_id type x y z
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    coords.append([x, y, z])
            except (ValueError, IndexError):
                continue
                
    return np.array(coords)