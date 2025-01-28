#!/usr/bin/env python3
import numpy as np
import os
import sys

def combine_results(size, checkpoint_dir="checkpoints", output_file="RMSD_Ncontact_data_final.txt"):
    """
    Concatenate each rank's results in ascending rank order (assuming contiguous frames).
    
    Parameters
    ----------
    size : int
        The total number of ranks used in the MPI job.
    checkpoint_dir : str
        Path to the directory containing results_rank_<rank>.npy files.
    output_file : str
        Name of the final output text file for RMSD & Ncontact.
    """
    all_data = []

    # Load each rank's file in ascending rank order
    for rank in range(size):
        fpath = os.path.join(checkpoint_dir, f"results_rank_{rank}.npy")
        if os.path.exists(fpath):
            data = np.load(fpath, allow_pickle=True)
            # 'data' should have shape (n_frames_for_this_rank, 2): [RMSD, Contacts]
            all_data.append(data)
            print(f"Loaded {len(data)} rows from rank {rank}")
        else:
            print(f"Warning: file not found for rank {rank}: {fpath}")
    
    # If no data found, exit
    if not all_data:
        print("No rank files were found. Nothing to combine.")
        return
    
    # Combine all arrays
    combined_results = np.concatenate(all_data, axis=0)
    print(f"Total combined frames: {len(combined_results)}")
    
    # Save combined data to a .npy file
    np.save("combined_results.npy", combined_results)
    
    # Also save to a text file
    np.savetxt(
        output_file,
        combined_results,
        header="RMSD Contacts",  # Column labels
        comments='#'
    )
    
    print(f"Final combined results saved to '{output_file}' and 'combined_results.npy'")

if __name__ == "__main__":
    """
    Usage:
        python combine_script.py <size> [checkpoint_dir] [output_file]

    Example:
        python combine_script.py 32
        python combine_script.py 32 my_checkpoints RMSD_contacts_final.txt
    """

    # We expect the first CLI argument to be the number of ranks
    if len(sys.argv) < 2:
        print("Usage: python combine_script.py <size> [checkpoint_dir] [output_file]")
        sys.exit(1)
    
    size = int(sys.argv[1])
    checkpoint_dir = "checkpoints"
    output_file = "RMSD_Ncontact_data_final.txt"

    # Optional second argument: custom checkpoint directory
    if len(sys.argv) >= 3:
        checkpoint_dir = sys.argv[2]
    # Optional third argument: custom output file name
    if len(sys.argv) >= 4:
        output_file = sys.argv[3]

    # Call the function to combine
    combine_results(size, checkpoint_dir, output_file)
