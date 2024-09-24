import os
import glob
import numpy as np
import pandas as pd
import torch
import logging

# Import your Chronset functions
from chronset_integration import detect_precise_onsets  # Ensure this function exists

# Optional: Import librosa if your Chronset implementation requires it
import librosa

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Runs a smokescreen test for the Chronset onset detection pipeline.
    Processes a few .wav files, detects onsets, and saves the results.
    """
    
    # -----------------------------
    # 1. Setup and Configuration
    # -----------------------------
    
    # Define whether to use GPU or CPU
    use_gpu = False
    device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')
    
    # Define input and output directories
    in_dir = 'dependencies/annotation_gauntlet/session_0'  # Adjust as needed
    output_dir = 'test/chronset_out/audio/'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created output directory: {output_dir}')
    else:
        print(f'Output directory already exists: {output_dir}')
    
    # -----------------------------
    # 2. Process Each .wav File
    # -----------------------------
    
    # Find all .wav files in the input directory
    wav_files = glob.glob(os.path.join(in_dir, '*.wav'))
    print(f'Found {len(wav_files)} .wav files in {in_dir}')
    
    if not wav_files:
        print('No .wav files found. Exiting.')
        return
    
    for wav_path in wav_files:
        try:
            print(f'\nProcessing file: {wav_path}')
            
            # Extract base name without extension
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            
            # -------------------------
            # a. Load Audio File
            # -------------------------
            
            # Load audio using librosa
            audio, sr = librosa.load(wav_path, sr=16000)  # Ensure sample rate matches Chronset's expectation
            print(f'Loaded audio: {wav_path} with sample rate: {sr} Hz')
            
            # -------------------------
            # b. Save Audio as .npy
            # -------------------------
            
            npy_filename = f'{base_name}.npy'
            npy_path = os.path.join(output_dir, npy_filename)
            np.save(npy_path, audio)
            print(f'Saved audio as .npy: {npy_path}')
            
            # -------------------------
            # c. Detect Precise Onsets
            # -------------------------
            
            # Detect onsets using Chronset
            onsets = detect_precise_onsets(wav_path, sr=sr, hop_length=160, energy_threshold=0.6)  # Pass file path
            print(f'Detected onsets (seconds): {onsets}')
            
            # -------------------------
            # d. Save Onsets to CSV
            # -------------------------
            
            csv_filename = f'{base_name}_onsets.csv'
            csv_path = os.path.join(output_dir, csv_filename)
            
            # Create a DataFrame for onsets
            df_onsets = pd.DataFrame({'onset_time_sec': onsets})
            df_onsets.to_csv(csv_path, index=False)
            print(f'Saved onsets to CSV: {csv_path}')
            
            # -------------------------
            # e. (Optional) Plot and Save Figure
            # -------------------------
            
            # If you have a plotting function, uncomment and use it
            # plot_basename = os.path.join(output_dir, base_name)
            # SaveFig(plot_basename)
            
        except Exception as e:
            print(f'Error processing {wav_path}: {e}')
            continue  # Proceed to the next file
    
    print('\nChronset smokescreen test completed.')

if __name__ == '__main__':
    main()
