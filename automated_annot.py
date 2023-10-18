import os
import sys
import whisperx
import gc 
import string
import pandas as pd

device = "cpu" 
batch_size = 16 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# Check if the directory path is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python automated_annot.py <directory>")
    sys.exit(1)

# The directory path provided as a command-line argument
directory = sys.argv[1]

# Check if the provided directory path exists
if not os.path.exists(directory):
    print(f"The directory {directory} does not exist.")
    sys.exit(1)

# Check if the provided path is a directory
if not os.path.isdir(directory):
    print(f"The path {directory} is not a directory.")
    sys.exit(1)
    
# create a subdirectory to store all results
out_dir = os.path.join(directory, 'whisperx_out')
os.makedirs(out_dir, exist_ok=True)

# List all .wav files in the directory
wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]

# Print the .wav files
if wav_files:
    print("WAV files found:")
    for file in wav_files:
        print("\n\nProcessing {}...".format(file))
        
        # define audio file
        filepath = os.path.join(directory, file)
        audio = whisperx.load_audio(filepath)
        result = model.transcribe(audio, batch_size=batch_size)
        #print(result["segments"]) # before alignment
        
        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del model
        
        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        #print(result["segments"]) # after alignment
        
        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

        words, onsets, confidence = [], [], []
        
        for segment in result["segments"]:
        
            word_list = segment['words']
            for element in word_list:
                #print(element)
                word = element['word']
                onset_time = element['start']
                prob = element['score']
            
                newword = word.translate(str.maketrans('', '', string.punctuation)).upper()
                words.append(newword); onsets.append(int(onset_time * 1000)); confidence.append(prob)
                #print(newword, int(onset_time * 1000), prob)

        # Create a DataFrame from the three lists
        df = pd.DataFrame({'Word': words, 'Onset': onsets, 'Probability': confidence})
        
        # Print the DataFrame (optional, for verification)
        #print(df)
        
        # Save the DataFrame to a CSV file
        output_name = os.path.splitext(file)[0] + ".csv"
        savepath = os.path.join(out_dir, output_name)
        df.to_csv(savepath, index=False)