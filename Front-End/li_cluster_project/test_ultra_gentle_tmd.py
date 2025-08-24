#!/usr/bin/env python3

import os
import sys
import django
from pathlib import Path

# Add the Django project to the Python path
sys.path.append('/Users/zhiyuan/Documents/HackthonAI4Sci/code/Front-End/li_cluster_project')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'li_cluster_project.settings')
django.setup()

from visualizer.lammps_simulator import LAMMPSSimulator
import numpy as np

def test_ultra_gentle_tmd():
    """Test the ultra-gentle 4-stage TMD implementation"""
    
    # Initialize simulator
    simulator = LAMMPSSimulator()
    
    # Define initial coordinates (Li3 linear structure)
    initial_coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0], 
        [4.0, 0.0, 0.0]
    ])
    
    # Target structure path
    target_pdb_path = "/Users/zhiyuan/Documents/HackthonAI4Sci/code/GNN_and_lammps_for_LiCluster_transformation/lammps_project/media/simulations/inputs/targets/Li3_stable.pdb"
    
    print("=== Ultra-Gentle 4-Stage TMD Test ===")
    print(f"Initial coords: {initial_coords}")
    print(f"Target PDB: {target_pdb_path}")
    
    # Run simulation with ultra-gentle parameters
    job_id = "ultra_gentle_tmd_test"
    
    try:
        result_path = simulator.run_simulation(
            initial_coords=initial_coords,
            target_pdb_path=target_pdb_path, 
            job_id=job_id
        )
        
        print(f"\n✅ Simulation completed successfully!")
        print(f"📁 Result trajectory: {result_path}")
        
        # Check trajectory file
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                lines = f.readlines()
            
            # Count frames
            frame_count = 0
            for line in lines:
                if line.strip().isdigit() and int(line.strip()) == 3:
                    frame_count += 1
            
            print(f"📊 Trajectory contains {frame_count} frames")
            
            # Show first few frames
            print(f"\n🔬 First frame coordinates:")
            frame_lines = []
            in_first_frame = False
            for i, line in enumerate(lines[:10]):
                if line.strip() == "3":
                    in_first_frame = True
                    continue
                elif in_first_frame and line.startswith("Li"):
                    parts = line.strip().split()
                    coords = [float(x) for x in parts[1:4]]
                    print(f"  Li: {coords}")
                    if len(frame_lines) == 2:
                        break
                    frame_lines.append(line)
            
        else:
            print(f"❌ Trajectory file not found at {result_path}")
            
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        
        # Check for log file
        log_path = f"/Users/zhiyuan/Documents/HackthonAI4Sci/code/Front-End/li_cluster_project/media/simulations/outputs/{job_id}/simulation.log"
        if os.path.exists(log_path):
            print(f"\n📋 Checking simulation log:")
            with open(log_path, 'r') as f:
                log_content = f.read()
            print(log_content[-1000:])  # Last 1000 characters

if __name__ == "__main__":
    test_ultra_gentle_tmd()