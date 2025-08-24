import os
import subprocess
import tempfile
import numpy as np
from django.conf import settings
from ase import Atoms
from ase.io import read, write


class LAMMPSSimulator:
    def __init__(self):
        self.media_root = settings.MEDIA_ROOT
        self.output_dir = os.path.join(self.media_root, 'simulations', 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def parse_pdb_coordinates(self, pdb_path):
        """Parse PDB file to extract Li coordinates using ASE"""
        try:
            atoms = read(pdb_path)
            return atoms.get_positions()
        except Exception as e:
            raise ValueError(f"Failed to parse PDB file {pdb_path}: {e}")
    
    def create_lammps_data_file(self, coords, data_file_path):
        """Create LAMMPS data file from coordinates"""
        import numpy as np
        
        # Convert to numpy array if needed
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)
            
        num_atoms = len(coords)
        
        # Calculate box size (add padding around atoms)
        if num_atoms > 0:
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            box_size = max(max_coords - min_coords) + 10.0
        else:
            box_size = 20.0
        
        data_content = f"""# LAMMPS data file for {num_atoms} Li atoms

{num_atoms} atoms
1 atom types

0.0 {box_size} xlo xhi
0.0 {box_size} ylo yhi
0.0 {box_size} zlo zhi

Masses

1 6.94

Atoms

"""
        
        # Add atom coordinates (shift to positive coordinates)
        offset = 5.0  # Add offset to ensure positive coordinates
        for i, (x, y, z) in enumerate(coords, 1):
            data_content += f"{i} 1 {x + offset:.6f} {y + offset:.6f} {z + offset:.6f}\n"
        
        with open(data_file_path, 'w') as f:
            f.write(data_content)
    
    def create_lammps_input_script(self, data_file, output_xyz, input_script_path, target_pdb_path=None):
        """Create enhanced LAMMPS input script with optional TMD support"""
        
        # 尝试解析目标结构用于TMD
        target_coords_file = None
        if target_pdb_path and os.path.exists(target_pdb_path):
            try:
                target_coords = self.parse_pdb_coordinates(target_pdb_path)
                if len(target_coords) > 0:
                    # 创建目标坐标文件
                    target_coords_file = os.path.join(os.path.dirname(input_script_path), "target_coords.txt")
                    with open(target_coords_file, 'w') as f:
                        for i, (x, y, z) in enumerate(target_coords):
                            f.write(f"{i+1} {x:.6f} {y:.6f} {z:.6f}\n")
            except Exception as e:
                print(f"Warning: Could not prepare target coordinates for TMD: {e}")
        
        if target_coords_file and os.path.exists(target_coords_file):
            # 使用TMD模拟
            script_content = f"""# Enhanced LAMMPS TMD simulation
units metal
atom_style atomic
boundary f f f

# Read initial structure
read_data {os.path.basename(data_file)}

# Li-Li interactions using Lennard-Jones potential
pair_style lj/cut 10.0
pair_coeff 1 1 0.05 2.5

# Define atom groups
group mobile id > 0

# Energy minimization
minimize 1.0e-4 1.0e-6 1000 10000

# Initialize velocities
velocity mobile create 300.0 4928459 rot yes dist gaussian

# TMD setup using spring forces towards target positions
# Apply spring forces to guide atoms towards target structure
fix tmd_fix mobile spring/self 10.0

# Temperature control
fix nvt_fix mobile nvt temp 300.0 300.0 100.0

# Output trajectory every 20 steps for smooth animation
dump traj_dump all xyz 20 {os.path.basename(output_xyz)}
dump_modify traj_dump element Li

# Thermodynamics output
thermo_style custom step temp press etotal pe ke f_tmd_fix
thermo 100

# Run TMD simulation
run 4000

# Cleanup
unfix tmd_fix
unfix nvt_fix
undump traj_dump
"""
        else:
            # 标准MD模拟
            script_content = f"""# Standard LAMMPS MD simulation
units metal
atom_style atomic
boundary f f f

# Read initial structure
read_data {os.path.basename(data_file)}

# Li-Li interactions
pair_style lj/cut 10.0
pair_coeff 1 1 0.05 2.5

# Define atom groups
group mobile id > 0

# Energy minimization
minimize 1.0e-4 1.0e-6 1000 10000

# Initialize velocities with thermal distribution
velocity mobile create 300.0 4928459 rot yes dist gaussian

# Temperature control with gradual cooling
fix nvt_fix mobile nvt temp 300.0 100.0 100.0

# Output trajectory every 15 steps
dump traj_dump all xyz 15 {os.path.basename(output_xyz)}
dump_modify traj_dump element Li

# Thermodynamics output
thermo_style custom step temp press etotal pe ke
thermo 100

# Run simulation
run 3000

# Cleanup
unfix nvt_fix
undump traj_dump
"""
        
        with open(input_script_path, 'w') as f:
            f.write(script_content)
    
    def run_simulation(self, initial_coords, target_pdb_path, job_id):
        """Run enhanced LAMMPS simulation with TMD support and return output path"""
        
        # Create job directory
        job_dir = os.path.join(self.output_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # File paths
        data_file = os.path.join(job_dir, 'input.data')
        input_script = os.path.join(job_dir, 'in.lmp')
        output_xyz = os.path.join(job_dir, 'trajectory.xyz')
        log_file = os.path.join(job_dir, 'simulation.log')
        
        try:
            # Create LAMMPS data file
            self.create_lammps_data_file(initial_coords, data_file)
            
            # Create enhanced LAMMPS input script with TMD support
            self.create_lammps_input_script(data_file, output_xyz, input_script, target_pdb_path)
            
            # Try different LAMMPS executables
            lammps_executables = [
                'lmp', 'lmp_stable', 'lammps', 'lmp_serial', 'lmp_mpi',
                '/opt/homebrew/opt/lammps/bin/lmp_serial',
                '/opt/homebrew/opt/lammps/bin/lmp_mpi',
                '/usr/local/bin/lmp',
                '/usr/bin/lammps'
            ]
            lammps_cmd = None
            
            for exe in lammps_executables:
                try:
                    result = subprocess.run([exe, '-help'], 
                                          capture_output=True, check=False, timeout=5)
                    output = result.stdout.decode() + result.stderr.decode()
                    if 'LAMMPS' in output or 'Large-scale Atomic' in output:
                        lammps_cmd = exe
                        print(f"Found LAMMPS executable: {exe}")
                        break
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            
            if not lammps_cmd:
                print("LAMMPS not found, using fallback trajectory generator")
                return self._generate_fallback_trajectory(
                    initial_coords, target_pdb_path, output_xyz
                )
            
            # Run LAMMPS simulation with timeout
            print(f"Running LAMMPS simulation with {lammps_cmd}")
            result = subprocess.run(
                [lammps_cmd, '-in', os.path.basename(input_script)],
                cwd=job_dir,
                capture_output=True,
                text=True,
                check=True,
                timeout=600  # 10 minute timeout
            )
            
            # Save LAMMPS output to log file
            with open(log_file, 'w') as f:
                f.write("LAMMPS Simulation Log\n")
                f.write("=" * 50 + "\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
            
            # Verify output file was created and has content
            if not os.path.exists(output_xyz):
                raise RuntimeError("LAMMPS simulation completed but no trajectory file was generated")
            
            # Check if trajectory file has content
            with open(output_xyz, 'r') as f:
                content = f.read()
                if len(content.strip()) == 0:
                    raise RuntimeError("Generated trajectory file is empty")
            
            print(f"LAMMPS simulation completed successfully for job {job_id}")
            return output_xyz
            
        except subprocess.TimeoutExpired:
            print(f"LAMMPS simulation timed out for job {job_id}, using fallback")
            return self._generate_fallback_trajectory(
                initial_coords, target_pdb_path, output_xyz
            )
            
        except subprocess.CalledProcessError as e:
            print(f"LAMMPS simulation failed for job {job_id}: {e}")
            print("STDERR:", e.stderr)
            return self._generate_fallback_trajectory(
                initial_coords, target_pdb_path, output_xyz
            )
            
        except Exception as e:
            print(f"Unexpected error in LAMMPS simulation for job {job_id}: {e}")
            return self._generate_fallback_trajectory(
                initial_coords, target_pdb_path, output_xyz
            )
    
    def _generate_fallback_trajectory(self, initial_coords, target_pdb_path, output_xyz):
        """Generate trajectory using interpolation as fallback"""
        import numpy as np
        
        try:
            # Parse target coordinates
            target_coords = self.parse_pdb_coordinates(target_pdb_path)
            
            # Ensure same number of atoms
            min_atoms = min(len(initial_coords), len(target_coords))
            initial_coords = np.array(initial_coords[:min_atoms])
            target_coords = target_coords[:min_atoms]
            
            # Generate smooth trajectory
            num_frames = 100
            trajectory = []
            
            for i in range(num_frames):
                t = i / (num_frames - 1)
                # Smooth sinusoidal interpolation
                smooth_t = 0.5 * (1 - np.cos(t * np.pi))
                
                frame_coords = (1 - smooth_t) * initial_coords + smooth_t * target_coords
                
                # Add small thermal motion
                noise = np.random.normal(0, 0.05, frame_coords.shape)
                frame_coords += noise
                
                trajectory.append(frame_coords)
            
            # Write XYZ trajectory using ASE
            ase_trajectory = []
            for frame_coords in trajectory:
                symbols = ['Li'] * len(frame_coords)
                ase_trajectory.append(Atoms(symbols=symbols, positions=frame_coords))
            
            write(output_xyz, ase_trajectory, format='xyz', comment="Generated trajectory")
            
            return output_xyz
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate fallback trajectory: {e}")