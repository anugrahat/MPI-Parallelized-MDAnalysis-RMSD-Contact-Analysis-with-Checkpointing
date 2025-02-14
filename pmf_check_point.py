#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np
import MDAnalysis as mda
from datetime import datetime
import os
import json
import pathlib
import sys

def ensure_directory(directory):
    """Ensure directory exists with proper permissions."""
    path = pathlib.Path(directory)
    if not path.exists():
        try:
            path.mkdir(parents=True, mode=0o777)
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            return False
    return True

def calculate_frame_data(ts, ref_positions, protein, ligand, cutoff=6.0):
    """Calculate RMSD and contact count for a single frame."""
    try:
        if ts.dimensions is not None and all(x > 0 for x in ts.dimensions[:3]):
            rmsd = np.sqrt(((ligand.positions - ref_positions) ** 2).sum(axis=1).mean())
            distance_matrix = np.linalg.norm(
                protein.positions[:, np.newaxis, :] - ligand.positions[np.newaxis, :, :], 
                axis=-1
            )
            contacts = np.sum(distance_matrix < cutoff)
            return rmsd, contacts
    except Exception as e:
        print(f"Error processing frame {ts.frame}: {str(e)}")
    return None, None

def save_checkpoint(rank, current_frame, results, checkpoint_dir="checkpoints"):
    """Save intermediate results and progress for this rank."""
    try:
        if not ensure_directory(checkpoint_dir):
            return
        if not results:
            return

        results_file = os.path.join(checkpoint_dir, f"results_rank_{rank}.npy")
        progress_file = os.path.join(checkpoint_dir, f"progress_rank_{rank}.json")
        np.save(results_file, np.array(results))

        progress = {
            "last_completed_frame": current_frame,
            "timestamp": datetime.now().isoformat(),
            "n_results": len(results)
        }

        with open(progress_file, 'w') as f:
            json.dump(progress, f)
            f.flush()
            os.fsync(f.fileno())

        print(f"Rank {rank}: Checkpoint saved at frame {current_frame} ({len(results)} frames processed)")
        sys.stdout.flush()

    except Exception as e:
        print(f"Rank {rank}: Error in save_checkpoint: {e}")
        sys.stdout.flush()

def load_checkpoint(rank, start_frame, checkpoint_dir="checkpoints"):
    """Load checkpoint for this rank if it exists."""
    results_file = os.path.join(checkpoint_dir, f"results_rank_{rank}.npy")
    progress_file = os.path.join(checkpoint_dir, f"progress_rank_{rank}.json")

    if os.path.exists(results_file) and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)

            last_frame = progress["last_completed_frame"]

            if last_frame >= start_frame:
                results = list(np.load(results_file, allow_pickle=True))
                print(f"Rank {rank}: Found checkpoint - Resuming from frame {last_frame+1}")
                sys.stdout.flush()
                return results, last_frame + 1

        except Exception as e:
            print(f"Rank {rank}: Error loading checkpoint: {e}")
            sys.stdout.flush()

    print(f"Rank {rank}: Starting fresh from frame {start_frame}")
    sys.stdout.flush()
    return [], start_frame

def process_frames(topology_file, trajectory_file, reference_file, start_frame, end_frame, rank, checkpoint_interval=1000):
    """Process a range of frames with checkpointing."""
    
    results, current_frame = load_checkpoint(rank, start_frame)

   
    u = mda.Universe(topology_file, trajectory_file, guess_bonds=False, guess_angles=False)
    ref_universe = mda.Universe(topology_file, reference_file, guess_bonds=False, guess_angles=False)

    ref_ligand = ref_universe.select_atoms('resid 275 and not type H')
    ref_positions = ref_ligand.positions.copy()
    ligand = u.select_atoms('resid 275 and not type H')
    protein = u.select_atoms('protein and resid 1-273 and not type H')

    total_frames = end_frame - current_frame
    frames_processed = 0

    
    for ts in u.trajectory[current_frame:end_frame]:
        rmsd, contacts = calculate_frame_data(ts, ref_positions, protein, ligand)
        if rmsd is not None and contacts is not None:
            results.append((rmsd, contacts))
            frames_processed += 1

        if (ts.frame + 1) % checkpoint_interval == 0:
            save_checkpoint(rank, ts.frame, results)
            progress = (frames_processed / total_frames) * 100
            print(f"Rank {rank}: {progress:.1f}% complete ({frames_processed}/{total_frames} frames)")
            sys.stdout.flush()

    
    if results:
        save_checkpoint(rank, end_frame - 1, results)
        print(f"Rank {rank}: Completed {frames_processed} frames")
        sys.stdout.flush()

    return results

def analyze_trajectory_mpi(topology_file, trajectory_file, reference_file, output_file="RMSD_Ncontact_data_final.txt"):
    """Analyze trajectory using MPI with checkpointing and resume capability."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"Process {rank} of {size} initialized on {MPI.Get_processor_name()}")
    sys.stdout.flush()

    if rank == 0:
        print("\nStarting analysis...")
        sys.stdout.flush()
        u = mda.Universe(topology_file, trajectory_file, guess_bonds=False, guess_angles=False)
        total_frames = len(u.trajectory)
        print(f"Total frames: {total_frames}")
        sys.stdout.flush()
        del u
    else:
        total_frames = None

   
    total_frames = comm.bcast(total_frames, root=0)

   
    chunk_size = total_frames // size
    remainder = total_frames % size

    if rank < remainder:
        start_frame = rank * (chunk_size + 1)
        frames_this_process = chunk_size + 1
    else:
        start_frame = remainder * (chunk_size + 1) + (rank - remainder) * chunk_size
        frames_this_process = chunk_size

    end_frame = start_frame + frames_this_process

    print(f"Process {rank}: Assigned frames {start_frame} to {end_frame-1}")
    sys.stdout.flush()

   
    results = process_frames(
        topology_file,
        trajectory_file,
        reference_file,
        start_frame,
        end_frame,
        rank
    )

   
    results_array = np.array(results)
    all_results = comm.gather(results_array, root=0)

def combine_results(size, output_file="RMSD_Ncontact_data_final.txt", checkpoint_dir="checkpoints"):
    """Combine results from all ranks in correct order."""
    print("Combining results from all ranks...")
    all_results = []
    
    
    for rank in range(size):
        results_file = os.path.join(checkpoint_dir, f"results_rank_{rank}.npy")
        if os.path.exists(results_file):
            try:
                rank_results = np.load(results_file, allow_pickle=True)
                all_results.append(rank_results)
                print(f"Loaded {len(rank_results)} frames from rank {rank}")
            except Exception as e:
                print(f"Error loading results from rank {rank}: {e}")
    
    if all_results:
      
        combined_results = np.vstack(all_results)
        print(f"Total frames combined: {len(combined_results)}")
        
       
        np.save("combined_results.npy", combined_results)
        np.savetxt(output_file, combined_results, header="RMSD Ncontact", comments='#')
        print(f"Results saved to {output_file} and combined_results.npy")
    else:
        print("No results found to combine")

if __name__ == "__main__":
    topology = '/expanse/lustre/scratch/anugrahat/temp_project/westpa/ParGaMD-main/common_files/ppar.parm7'
    trajectory = '/expanse/lustre/scratch/anugrahat/temp_project/westpa/ParGaMD-main/combined.nc'
    reference = '/expanse/lustre/scratch/anugrahat/temp_project/westpa/ParGaMD-main/common_files/200000th_frame.pdb'
    
    analyze_trajectory_mpi(topology, trajectory, reference)
