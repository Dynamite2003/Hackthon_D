from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponseServerError
from django.contrib import messages
from django.conf import settings
import subprocess
import os
import time
import re
import numpy as np

def parse_log_for_sites(logs):
    """
    使用正则表达式解析日志,提取Top-K位点信息。
    """
    sites = []
    # 正则表达式模式,用于捕获每个位点的信息
    pattern = re.compile(
        r"位点 #(?P<rank>\d+):\n"
        r"\s+坐标: (?P<coords>.*?)\sÅ\n"
        r"\s+预测能量: (?P<energy>[\d.-]+) eV",
        re.MULTILINE
    )
    
    matches = pattern.finditer(logs);
    for match in matches:
        site_data = match.groupdict()
        # 清理坐标字符串,移除可能存在的括号和多余的空格
        coords = site_data.get('coords', '')
        site_data['coords'] = coords.replace('(', '').replace(')', '').strip()
        sites.append(site_data)
    
    return sites

def parse_pdb_for_coords(pdb_path):
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('HETATM') or line.startswith('ATOM'):
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append(f"{x},{y},{z}")
                except (ValueError, IndexError):
                    continue
    return "\n".join(coords)

def create_coordinates_file(coords):
    """创建坐标文件（简单的逗号分隔格式）"""
    lines = []
    for x, y, z in coords:
        lines.append(f"{x:.6f},{y:.6f},{z:.6f}")
    return "\n".join(lines)

def create_lammps_data_file(coords):
    """创建LAMMPS数据文件(.data格式)"""
    num_atoms = len(coords)
    
    # 计算盒子尺寸以容纳所有坐标（包括负坐标）
    if num_atoms > 0:
        coords_array = np.array(coords)
        min_coords = coords_array.min(axis=0)
        max_coords = coords_array.max(axis=0)
        padding = 5.0
        
        xlo, xhi = min_coords[0] - padding, max_coords[0] + padding
        ylo, yhi = min_coords[1] - padding, max_coords[1] + padding
        zlo, zhi = min_coords[2] - padding, max_coords[2] + padding
    else:
        xlo, xhi = -10.0, 10.0
        ylo, yhi = -10.0, 10.0
        zlo, zhi = -10.0, 10.0
    
    # 生成LAMMPS数据文件内容
    content = f"""# LAMMPS data file for {num_atoms} Li atoms

{num_atoms} atoms
1 atom types

{xlo:.6f} {xhi:.6f} xlo xhi
{ylo:.6f} {yhi:.6f} ylo yhi
{zlo:.6f} {zhi:.6f} zlo zhi

Masses

1 6.94

Atoms

"""
    
    # 添加原子坐标（保持原始坐标）
    for i, (x, y, z) in enumerate(coords, 1):
        content += f"{i} 1 {x:.6f} {y:.6f} {z:.6f}\n"
    
    return content

def start_trajectory_simulation(initial_coords_file, target_pdb_path, job_id, trajectory_project_dir):
    """启动轨迹生成任务"""
    try:
        # 创建本地模拟任务记录
        from .models import SimulationJob
        job = SimulationJob.objects.create(
            id=job_id,
            structure_a_path=initial_coords_file,
            structure_b_path=target_pdb_path
        )
        
        # 轨迹生成器路径
        trajectory_generator_path = os.path.join(trajectory_project_dir, 'trajectory_generator.py')
        
        # 输出路径
        output_dir = os.path.join(trajectory_project_dir, 'media', 'simulations', 'outputs', str(job_id))
        os.makedirs(output_dir, exist_ok=True)
        output_xyz_path = os.path.join(output_dir, 'trajectory.xyz')
        
        # 构建命令来运行轨迹生成器
        command = [
            'python3', trajectory_generator_path,
            initial_coords_file,  # coords_file
            target_pdb_path,      # target_pdb
            output_xyz_path,      # output_xyz
            '--frames', '100'
        ]
        
        # 异步启动进程
        import subprocess
        process = subprocess.Popen(
            command,
            cwd=trajectory_project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 更新任务状态
        job.status = SimulationJob.Status.STARTED
        job.save()
        
        return {
            'success': True,
            'job_id': job_id,
            'process_id': process.pid,
            'output_path': output_xyz_path
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def start_lammps_simulation(initial_coords, target_pdb_path, job_id, lammps_project_dir):
    """启动LAMMPS模拟任务"""
    try:
        # 创建本地模拟任务记录
        from .models import SimulationJob
        job = SimulationJob.objects.create(
            id=job_id,
            structure_a_path="lammps_input",
            structure_b_path=target_pdb_path
        )
        
        # 创建LAMMPS数据文件
        lammps_data_content = create_lammps_data_file(initial_coords)
        
        # 输出路径
        output_dir = os.path.join(lammps_project_dir, 'media', 'simulations', 'outputs', str(job_id))
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存LAMMPS数据文件
        data_file_path = os.path.join(output_dir, 'input.data')
        with open(data_file_path, 'w') as f:
            f.write(lammps_data_content)
        
        # 复制目标PDB文件
        import shutil
        target_copy_path = os.path.join(output_dir, 'target.pdb')
        shutil.copy2(target_pdb_path, target_copy_path)
        
        # 使用LAMMPS模拟器
        from .lammps_simulator import LAMMPSSimulator
        simulator = LAMMPSSimulator()
        
        # 运行LAMMPS模拟
        output_xyz_path = simulator.run_simulation(
            initial_coords, 
            target_copy_path, 
            str(job_id)
        )
        
        if output_xyz_path and os.path.exists(output_xyz_path):
            # 更新任务状态
            job.status = SimulationJob.Status.SUCCESS
            job.output_trajectory_path = output_xyz_path
            job.save()
            
            return {
                'success': True,
                'job_id': job_id,
                'output_path': output_xyz_path
            }
        else:
            job.status = SimulationJob.Status.FAILURE
            job.save()
            return {
                'success': False,
                'error': 'LAMMPS simulation failed to generate trajectory'
            }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def wait_for_trajectory_completion(job_id, trajectory_project_dir, output_path, timeout=300):
    """等待轨迹生成完成并返回轨迹文件路径"""
    import time
    
    try:
        from .models import SimulationJob
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                job = SimulationJob.objects.get(id=job_id)
                
                # 检查输出文件是否存在
                if os.path.exists(output_path):
                    job.status = SimulationJob.Status.SUCCESS
                    job.output_trajectory_path = output_path
                    job.save()
                    return output_path
                
                time.sleep(2)  # 等待2秒后再检查
                
            except SimulationJob.DoesNotExist:
                return None
        
        # 超时处理
        try:
            job = SimulationJob.objects.get(id=job_id)
            job.status = SimulationJob.Status.FAILURE
            job.save()
        except SimulationJob.DoesNotExist:
            pass
            
        return None  # 超时
        
    except Exception as e:
        print(f"等待轨迹生成完成时出错: {e}")
        return None

def index(request):
    plot_html = None
    logs = ""
    parsed_sites = []
    download_path = request.GET.get('download_path')  # 从GET参数获取下载路径
    # 默认坐标,用于填充表单
    default_coords = (
        "-1.7138,-2.5707,2.1413\n"
        "-3.4276,-0.8569,0.4275\n"
        "0.0000,-0.8569,0.4275\n"
        "0.0000,-4.2826,0.4275\n"
        "1.7138,0.8569,-1.2844\n"
        "3.4276,2.5707,0.4275\n"
        "0.0000,2.5707,0.4275\n"
        "0.0000,2.5707,-2.9982"
    )
    # 在函数开始时设置一个变量来持有坐标,默认为 default_coords
    coords_to_display = default_coords

    if request.method == 'POST':
        if 'start_simulation' in request.POST:
            # 自动从输入框读取坐标
            initial_coords_text = request.POST.get('coords_text', default_coords)
            selected_site_coords = request.POST.get('selected_site_coords', '')
            
            # 解析初始坐标
            initial_lines = initial_coords_text.strip().split('\n')
            initial_coords = []
            for line in initial_lines:
                line = line.strip()
                if line:
                    try:
                        x, y, z = map(float, line.split(','))
                        initial_coords.append((x, y, z))
                    except ValueError:
                        pass # Skip malformed lines
            
            # 如果有选择的位点坐标，添加到初始坐标中
            if selected_site_coords:
                try:
                    x, y, z = map(float, selected_site_coords.split(','))
                    initial_coords.append((x, y, z))
                except ValueError:
                    pass
            
            num_atoms = len(initial_coords)
            if num_atoms == 0:
                messages.error(request, "No valid coordinates provided.")
                return redirect('index')
            
            # 构建坐标文本（用于传递给轨迹生成器）
            coords_text = '\n'.join([f"{x},{y},{z}" for x, y, z in initial_coords])
            
            # 获取模拟方式选择
            simulation_method = request.POST.get('simulation_method', 'trajectory')

            # --- 根据选择的方式调用相应的模拟器 ---
            import tempfile
            import uuid
            import shutil
            
            # 使用当前Django项目的路径
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            trajectory_project_dir = current_dir  # 轨迹生成器项目路径与当前项目相同
            
            try:
                # 1. 找到目标结构文件
                target_pdb_path = os.path.join(
                    current_dir, 
                    'media', 'simulations', 'inputs', 'targets',
                    f'Li{num_atoms}_stable.pdb'
                )
                
                if not os.path.exists(target_pdb_path):
                    messages.error(request, f"目标结构文件不存在: Li{num_atoms}_stable.pdb")
                    return redirect('index')
                
                # 2. 生成任务ID
                job_id = str(uuid.uuid4())
                
                # 3. 根据选择的方式启动模拟
                if simulation_method == 'lammps':
                    # LAMMPS模拟
                    simulation_result = start_lammps_simulation(
                        initial_coords,
                        target_pdb_path, 
                        job_id,
                        trajectory_project_dir
                    )
                else:
                    # 轨迹生成器模拟
                    temp_coords_file = os.path.join(
                        trajectory_project_dir, 
                        'media', 'simulations', 'inputs',
                        f'temp_coords_{job_id}.txt'
                    )
                    
                    with open(temp_coords_file, 'w') as f:
                        f.write(coords_text)
                    
                    simulation_result = start_trajectory_simulation(
                        temp_coords_file, 
                        target_pdb_path, 
                        job_id,
                        trajectory_project_dir
                    )
                
                if simulation_result['success']:
                    method_name = "LAMMPS模拟" if simulation_method == 'lammps' else "轨迹生成"
                    messages.success(request, f"{method_name}已启动，任务ID: {simulation_result['job_id']}")
                    
                    # 等待模拟完成并获取结果
                    if simulation_method == 'lammps':
                        # LAMMPS模拟直接返回结果
                        trajectory_path = simulation_result['output_path']
                    else:
                        # 轨迹生成器需要等待
                        trajectory_path = wait_for_trajectory_completion(
                            simulation_result['job_id'], 
                            trajectory_project_dir,
                            simulation_result['output_path']
                        )
                    
                    if trajectory_path:
                        # 清理临时文件（仅对轨迹生成器）
                        if simulation_method == 'trajectory':
                            try:
                                os.remove(temp_coords_file)
                            except:
                                pass
                        
                        # 复制文件到Django的media目录以便下载
                        from django.conf import settings
                        import shutil
                        django_media_dir = settings.MEDIA_ROOT
                        os.makedirs(django_media_dir, exist_ok=True)
                        
                        # 创建一个唯一的文件名
                        trajectory_filename = f"trajectory_{job_id}.xyz"
                        django_trajectory_path = os.path.join(django_media_dir, trajectory_filename)
                        
                        shutil.copy2(trajectory_path, django_trajectory_path)
                        download_path = f"/media/{trajectory_filename}"
                        
                        redirect_url = reverse('index')
                        redirect_url += f"?download_path={download_path}"
                        return redirect(redirect_url)
                    else:
                        messages.error(request, "轨迹生成完成但未找到轨迹文件")
                else:
                    messages.error(request, f"启动轨迹生成失败: {simulation_result['error']}")
                
            except Exception as e:
                messages.error(request, f"轨迹生成过程出错: {str(e)}")
            
            return redirect('index')

        elif 'coords_text' in request.POST or request.POST.get('input_type') == 'atom_count':
            # --- 从表单提取数据 ---
            input_type = request.POST.get('input_type', 'coords')
            coords_text = ""

            if input_type == 'atom_count':
                num_atoms = request.POST.get('num_atoms')
                if not num_atoms:
                    messages.error(request, "请输入原子数。")
                    return redirect('index')
                
                # 使用当前Django项目的路径
                current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                pdb_file_path = os.path.join(current_dir, 'media', 'simulations', 'inputs', 'targets', f'Li{num_atoms}_stable.pdb')

                if not os.path.exists(pdb_file_path):
                    messages.error(request, f"未找到 {num_atoms} 个原子的稳定结构文件。")
                    return redirect('index')
                
                try:
                    coords_text = parse_pdb_for_coords(pdb_file_path)
                    if not coords_text:
                        raise ValueError("PDB文件中未找到有效的原子坐标。")
                except Exception as e:
                    messages.error(request, f"解析PDB文件时出错: {e}")
                    return redirect('index')
            else:
                coords_text = request.POST.get('coords_text', default_coords)
            # 在处理后,更新要显示的坐标
            coords_to_display = coords_text
            search_strategy = request.POST.get('search_strategy', 'combined')
            top_k = request.POST.get('top_k', 1)
            visualization_type = request.POST.get('visualization_type', 'space_filling')

            # --- 运行后端脚本 ---
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            code_root = os.path.dirname(project_root)

            coords_filename = os.path.join(code_root, "django_input_coords.txt")
            output_html_path = os.path.join(code_root, "test_advanced_molecular_viz.html")
            script_path = os.path.join(code_root, "run_predict_sites.py")

            with open(coords_filename, "w") as f:
                f.write(coords_text.strip())

            command = [
                "python3", script_path,
                "--file", coords_filename,
                "--strategy", search_strategy,
                "--top-k", str(top_k),
                "--save-plot", output_html_path,
                "--advanced-renderer",
                "--visualization-type", visualization_type
            ]

            logs = f"▶️ Working Directory: {code_root}\n"
            logs += f"▶️ Executing command: {' '.join(command)}\n\n"

            try:
                process = subprocess.run(
                    command,
                    capture_output=True, text=True, check=True, encoding='utf-8',
                    cwd=code_root
                )
                logs += "✅ Command successful.\n"
                logs += "--- STDOUT ---\n" + process.stdout
                logs += "--- STDERR ---\n" + process.stderr

                # 解析日志以提取位点信息
                parsed_sites = parse_log_for_sites(logs)

                # 定义与 visualizer 一致的调色板,并为每个位点添加颜色属性
                color_palette = {
                    '1': 'gold',    
                    '2': '#0077BB', 
                    '3': '#EE7733', 
                    '4': '#009988', 
                    '5': '#CC311', 
                }
                default_color = '#BBBBBB'

                for site in parsed_sites:
                    site['color'] = color_palette.get(site['rank'], default_color)

                # 重试逻辑以查找文件
                file_found = False
                max_retries = 5
                retry_delay = 0.2

                for i in range(max_retries):
                    if os.path.exists(output_html_path):
                        file_found = True
                        logs += f"\n[Retry {i+1}/{max_retries}] File found."
                        break
                    logs += f"\n[Retry {i+1}/{max_retries}] File not found, waiting {retry_delay}s..."
                    time.sleep(retry_delay)

                if file_found:
                    with open(output_html_path, "r", encoding='utf-8') as f:
                        plot_html = f.read()
                    plot_html += f'<!-- {time.time()} -->'
                else:
                    logs += f"\n⚠️ Warning: Script finished, but output file was not found."

            except subprocess.CalledProcessError as e:
                logs += f"❌ Command failed with exit code {e.returncode}\n"
                logs += "--- STDOUT ---\n" + e.stdout
                logs += "--- STDERR ---\n" + e.stderr
            except Exception as e:
                logs += f"❌ An unknown error occurred: {str(e)}"
    
    context = {
        'plot_html': plot_html,
        'logs': logs,
        # 使用 coords_to_display 而不是硬编码的 default_coords
        'default_coords': coords_to_display,
        'parsed_sites': parsed_sites,
        'download_path': download_path,  # 将下载路径传递到模板
    }
    return render(request, 'visualizer/index.html', context)


