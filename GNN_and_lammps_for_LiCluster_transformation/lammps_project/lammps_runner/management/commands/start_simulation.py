import os
import sys
import tempfile
import uuid
from django.core.management.base import BaseCommand
from django.conf import settings
from lammps_runner.lammps_simulator import LAMMPSSimulator


class Command(BaseCommand):
    help = 'Start LAMMPS simulation with given coordinates'

    def add_arguments(self, parser):
        parser.add_argument('coords_file', type=str, help='Path to coordinates file')

    def handle(self, *args, **options):
        coords_file = options['coords_file']
        
        try:
            # Read coordinates from file
            with open(coords_file, 'r') as f:
                coords_text = f.read().strip()
            
            # Parse coordinates
            coords = []
            for line in coords_text.split('\n'):
                line = line.strip()
                if line:
                    try:
                        x, y, z = map(float, line.split(','))
                        coords.append((x, y, z))
                    except ValueError:
                        continue
            
            if not coords:
                self.stderr.write("No valid coordinates found")
                sys.exit(1)
            
            num_atoms = len(coords)
            
            # Find target structure
            target_pdb = os.path.join(
                settings.MEDIA_ROOT, 
                'simulations', 'inputs', 'targets', 
                f'Li{num_atoms}_stable.pdb'
            )
            
            if not os.path.exists(target_pdb):
                self.stderr.write(f"Target structure not found: {target_pdb}")
                sys.exit(1)
            
            # Create simulator
            simulator = LAMMPSSimulator()
            
            # Generate unique job ID
            job_id = str(uuid.uuid4())
            
            # Run simulation
            output_path = simulator.run_simulation(
                initial_coords=coords,
                target_pdb_path=target_pdb,
                job_id=job_id
            )
            
            # Output the path for the calling process
            self.stdout.write(output_path)
            self.stderr.write(f"Simulation completed successfully for {num_atoms} atoms")
            
        except Exception as e:
            self.stderr.write(f"Simulation failed: {str(e)}")
            sys.exit(1)