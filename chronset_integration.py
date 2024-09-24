import librosa
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_precise_onsets(audio_path: str, sr: int = 16000, hop_length: int = 10, energy_threshold: float = 0.6) -> List[float]:
    """
    Detects precise speech onsets using librosa.

    Parameters:
    - audio_path: Path to the audio file.
    - sr: Sampling rate.
    - hop_length: Number of samples between successive frames.
    - energy_threshold: Threshold for onset detection.

    Returns:
    - List of onset times in seconds.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                           hop_length=hop_length, backtrack=False,
                                           units='time', delta=energy_threshold)
        logging.info(f"Detected {len(onsets)} onsets in '{audio_path}'.")
        return onsets.tolist()
    except Exception as e:
        logging.error(f"Error detecting onsets in '{audio_path}': {e}")
        return []

def align_onsets_with_transcription(transcribed_words: List[dict], precise_onsets: List[float]) -> List[dict]:
    """
    Aligns precise onsets with transcribed words.

    Parameters:
    - transcribed_words: List of dictionaries containing word transcriptions and initial timings.
    - precise_onsets: List of precise onset times in seconds.

    Returns:
    - List of dictionaries with updated onset times.
    """
    aligned_words = []
    onset_idx = 0
    num_onsets = len(precise_onsets)

    for word_info in transcribed_words:
        if onset_idx < num_onsets:
            # Assign the next precise onset
            word_info['onset'] = int(precise_onsets[onset_idx] * 1000)  # Convert to milliseconds
            onset_idx += 1
        else:
            # If no precise onset left, retain the original onset
            word_info['onset'] = int(word_info.get('start', 0) * 1000)
        # Optionally, refine offset as well if needed
        word_info['offset'] = int(word_info.get('end', 0) * 1000)
        aligned_words.append(word_info)

    logging.info(f"Aligned {len(aligned_words)} words with precise onsets.")
    return aligned_words

def save_aligned_annotations(aligned_words: List[dict], save_path: str):
    """
    Saves the aligned words with updated onset times to a CSV file.

    Parameters:
    - aligned_words: List of dictionaries containing word annotations.
    - save_path: Path to save the CSV file.
    """
    try:
        df = pd.DataFrame(aligned_words)
        df = df[['word', 'onset', 'offset', 'score']]  # Adjust columns as needed
        df.rename(columns={'word': 'Word', 'onset': 'Onset', 'offset': 'Offset', 'score': 'Probability'}, inplace=True)
        df.to_csv(save_path, index=False)
        logging.info(f"Saved aligned annotations to '{save_path}'.")
    except Exception as e:
        logging.error(f"Error saving aligned annotations to '{save_path}': {e}")

def dynamic_parameter_tuning(audio_path: str, window_size: float = 1.0, step_size: float = 0.5, 
                             sr: int = 16000, base_hop_length: int = 160, base_energy_threshold: float = 0.6) -> List[float]:
    """
    Dynamically tunes hop_length and energy_threshold using a sliding window approach to detect precise onsets.
    
    Parameters:
    - audio_path: Path to the audio file.
    - window_size: Size of the sliding window in seconds.
    - step_size: Step size for the sliding window in seconds.
    - sr: Sampling rate.
    - base_hop_length: Base hop_length to start tuning.
    - base_energy_threshold: Base energy_threshold to start tuning.
    
    Returns:
    - List of dynamically tuned onset times in seconds.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        total_duration = librosa.get_duration(y=y, sr=sr)
        onsets = []
        
        # Slide the window across the audio
        for start in np.arange(0, total_duration - window_size, step_size):
            end = start + window_size
            y_segment = y[int(start * sr):int(end * sr)]
            
            # Analyze segment to adjust parameters
            rms = librosa.feature.rms(y=y_segment)
            avg_rms = np.mean(rms)
            
            # Example adjustment: Lower energy_threshold in quieter segments
            if avg_rms < 0.01:
                energy_threshold = base_energy_threshold + 0.1
                hop_length = base_hop_length // 2  # Increase temporal resolution
            elif avg_rms > 0.03:
                energy_threshold = base_energy_threshold - 0.1
                hop_length = base_hop_length  # Default
            else:
                energy_threshold = base_energy_threshold
                hop_length = base_hop_length
            
            # Ensure thresholds stay within reasonable bounds
            energy_threshold = max(0.2, min(energy_threshold, 0.8))
            hop_length = max(80, min(hop_length, 320))  # Example bounds
            
            onset_env = librosa.onset.onset_strength(y=y_segment, sr=sr, hop_length=hop_length)
            detected_onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                                         hop_length=hop_length, backtrack=False,
                                                         units='time', delta=energy_threshold)
            # Adjust onset times relative to the entire audio
            adjusted_onsets = detected_onsets + start
            onsets.extend(adjusted_onsets.tolist())
        
        # Remove duplicate onsets and sort
        onsets = sorted(list(set(onsets)))
        logging.info(f"Detected {len(onsets)} onsets with dynamic tuning in '{audio_path}'.")
        return onsets
    except Exception as e:
        logging.error(f"Error in dynamic parameter tuning for '{audio_path}': {e}")
        return []

def process_audio_file(audio_path: str, transcription: Dict, save_csv_path: str, 
                      window_size: float = 1.0, step_size: float = 0.5, 
                      base_hop_length: int = 160, base_energy_threshold: float = 0.6):
    """
    Processes a single audio file: dynamically tunes parameters, detects precise onsets, aligns with 
    transcription, and saves the results.

    Parameters:
    - audio_path: Path to the audio file.
    - transcription: Transcription dictionary from WhisperX.
    - save_csv_path: Path to save the aligned CSV annotations.
    - window_size: Sliding window size in seconds.
    - step_size: Sliding window step size in seconds.
    - base_hop_length: Base hop_length for onset detection.
    - base_energy_threshold: Base energy_threshold for onset detection.
    """
    transcribed_words = transcription.get('segments', [])
    transcribed_words_flat = [word for segment in transcribed_words for word in segment.get('words', [])]

    # Dynamically tune parameters and detect precise onsets
    precise_onsets = dynamic_parameter_tuning(
        audio_path, 
        window_size=window_size, 
        step_size=step_size, 
        sr=16000, 
        base_hop_length=base_hop_length, 
        base_energy_threshold=base_energy_threshold
    )

    # Align onsets with transcription
    aligned_words = align_onsets_with_transcription(transcribed_words_flat, precise_onsets)

    # Save the aligned annotations
    save_aligned_annotations(aligned_words, save_csv_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Chronset Integration for Precise Word Onset Detection')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to the audio file.')
    parser.add_argument('--transcription_json', type=str, required=True, help='Path to the transcription JSON file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the aligned CSV annotations.')
    args = parser.parse_args()

    # Load transcription
    try:
        with open(args.transcription_json, 'r') as f:
            transcription = json.load(f)
        logging.info(f"Loaded transcription from '{args.transcription_json}'.")
    except Exception as e:
        logging.error(f"Error loading transcription from '{args.transcription_json}': {e}")
        sys.exit(1)

    # Process the audio file
    process_audio_file(args.audio_path, transcription, args.save_path)
