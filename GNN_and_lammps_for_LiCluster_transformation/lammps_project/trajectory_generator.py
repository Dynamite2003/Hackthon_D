#!/usr/bin/env python3
"""
Simple trajectory generator to replace LAMMPS
Generates smooth transition trajectories between initial and target coordinates
---
Refactored to use the Atomic Simulation Environment (ASE) for professional data handling
while maintaining the original class and function structure.
"""

import numpy as np
import os
import argparse
from typing import List, Tuple, Union

# ASE is used for professional handling of atomic structures and I/O
try:
    from ase import Atoms
    from ase.io import read, write
except ImportError:
    print("Error: ASE not found. Please install it using 'pip install ase'")
    exit(1)


class TrajectoryGenerator:
    def __init__(self, element_symbol: str = "Li"):
        self.element_symbol = element_symbol
    
    def parse_coordinates(self, coords_text: str) -> np.ndarray:
        """
        Parse coordinate text into a numpy array.
        This function is kept for compatibility with the original custom format.
        """
        lines = coords_text.strip().split('\n')
        coords = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    x, y, z = map(float, line.split(','))
                    coords.append([x, y, z])
                except ValueError:
                    continue
        return np.array(coords)
    
    def parse_pdb_file(self, pdb_path: str) -> np.ndarray:
        """
        Parse PDB file to extract coordinates using ASE.
        Returns coordinates as a numpy array to maintain original function signature.
        """
        try:
            # ASE's reader is robust and handles PDB format professionally
            atoms = read(pdb_path)
            return atoms.get_positions()
        except FileNotFoundError:
            print(f"Warning: PDB file {pdb_path} not found")
            return np.array([])
        except Exception as e:
            print(f"Warning: Could not read PDB file {pdb_path} with ASE. Error: {e}")
            return np.array([])

    def generate_interpolated_trajectory(self, 
                                       initial_coords: np.ndarray, 
                                       target_coords: np.ndarray, 
                                       num_frames: int = 100) -> List[np.ndarray]:
        """Generate trajectory using simple linear interpolation."""
        if len(initial_coords) != len(target_coords):
            print(f"Warning: Initial coords ({len(initial_coords)}) != Target coords ({len(target_coords)})")
            min_atoms = min(len(initial_coords), len(target_coords))
            initial_coords = initial_coords[:min_atoms]
            target_coords = target_coords[:min_atoms]
        
        trajectory_coords = []
        for i in range(num_frames):
            t = i / (num_frames - 1)
            frame_coords = (1 - t) * initial_coords + t * target_coords
            trajectory_coords.append(frame_coords.copy())
        
        return trajectory_coords
    
    def generate_smooth_trajectory(self, 
                                 initial_coords: np.ndarray, 
                                 target_coords: np.ndarray, 
                                 num_frames: int = 100) -> List[np.ndarray]:
        """Generate trajectory using smooth sinusoidal interpolation."""
        if len(initial_coords) != len(target_coords):
            min_atoms = min(len(initial_coords), len(target_coords))
            initial_coords = initial_coords[:min_atoms]
            target_coords = target_coords[:min_atoms]
        
        trajectory_coords = []
        for i in range(num_frames):
            t = i / (num_frames - 1)
            # Professional sinusoidal ease-in-out function
            smooth_t = 0.5 * (1 - np.cos(t * np.pi))
            
            frame_coords = (1 - smooth_t) * initial_coords + smooth_t * target_coords
            trajectory_coords.append(frame_coords.copy())
        
        return trajectory_coords
    
    def add_thermal_motion(self, trajectory: List[np.ndarray], temperature: float = 0.1) -> List[np.ndarray]:
        """Add small random thermal motion to make trajectory more realistic."""
        for frame in trajectory:
            noise = np.random.normal(0, temperature, frame.shape)
            frame += noise
        return trajectory
    
    def write_xyz_trajectory(self, trajectory: List[np.ndarray], output_path: str):
        """
        Write trajectory to XYZ format file using ASE.
        This is far more robust than manual string formatting.
        """
        # Create a list of ASE Atoms objects from the raw coordinate trajectory
        ase_trajectory = []
        for frame_coords in trajectory:
            num_atoms = len(frame_coords)
            symbols = [self.element_symbol] * num_atoms
            ase_trajectory.append(Atoms(symbols=symbols, positions=frame_coords))
        
        # ASE's `write` function handles the entire trajectory correctly
        write(output_path, ase_trajectory, format='xyz', comment=f"Generated trajectory")

    def generate_trajectory_from_coords(self, 
                                      initial_coords_text: str,
                                      target_pdb_path: str,
                                      output_path: str,
                                      num_frames: int = 100,
                                      add_thermal: bool = True):
        """Main function to generate trajectory from coordinate inputs."""
        initial_coords = self.parse_coordinates(initial_coords_text)
        if len(initial_coords) == 0:
            raise ValueError("No valid initial coordinates found")
        
        target_coords = self.parse_pdb_file(target_pdb_path)
        if len(target_coords) == 0:
            print("No target coordinates found, using displaced initial coordinates")
            target_coords = initial_coords + np.random.normal(0, 0.5, initial_coords.shape)
        
        print(f"Initial atoms: {len(initial_coords)}")
        print(f"Target atoms: {len(target_coords)}")
        
        trajectory = self.generate_smooth_trajectory(initial_coords, target_coords, num_frames)
        
        if add_thermal:
            trajectory = self.add_thermal_motion(trajectory, temperature=0.05)
        
        self.write_xyz_trajectory(trajectory, output_path)
        print(f"Generated trajectory with {len(trajectory)} frames")
        print(f"Output written to: {output_path}")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Generate molecular trajectory')
    parser.add_argument('coords_file', help='File containing initial coordinates')
    parser.add_argument('target_pdb', help='Target PDB file')
    parser.add_argument('output_xyz', help='Output XYZ trajectory file')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames')
    parser.add_argument('--no-thermal', action='store_true', help='Disable thermal motion')
    
    args = parser.parse_args()
    
    try:
        with open(args.coords_file, 'r') as f:
            coords_text = f.read()
    except FileNotFoundError:
        print(f"Error: Coordinates file {args.coords_file} not found")
        return 1
    
    generator = TrajectoryGenerator()
    try:
        generator.generate_trajectory_from_coords(
            coords_text,
            args.target_pdb,
            args.output_xyz,
            num_frames=args.frames,
            add_thermal=not args.no_thermal
        )
        return 0
    except Exception as e:
        print(f"Error generating trajectory: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())