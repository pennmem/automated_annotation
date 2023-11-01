import os
import sys
import whisperx
import gc 
import string
import pandas as pd
import json
import argparse
# from dask.distributed import print

from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH, DEFAULT_ALIGN_MODELS_HF, LANGUAGES_WITHOUT_SPACES


def get_model_dictionary(language_code='en', model_name=None, model_dir=None):
    # adapted from https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py
    import torchaudio
    
    if model_name is None:
        # use default model
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            print(f"There is no default alignment model set for this language ({language_code}).\
                Please find a wav2vec2.0 model finetuned on this language in https://huggingface.co/models, then pass the model name in --align_model [MODEL_NAME]")
            raise ValueError(f"No default align-model for language: {language_code}")

    if model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]
        # align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device)
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    # else:
    #     try:
    #         processor = Wav2Vec2Processor.from_pretrained(model_name)
    #         align_model = Wav2Vec2ForCTC.from_pretrained(model_name)
    #     except Exception as e:
    #         print(e)
    #         print(f"Error loading model from huggingface, check https://huggingface.co/models for finetuned wav2vec2.0 models")
    #         raise ValueError(f'The chosen align_model "{model_name}" could not be found in huggingface (https://huggingface.co/models) or torchaudio (https://pytorch.org/audio/stable/pipelines.html#id14)')
    #     pipeline_type = "huggingface"
    #     align_model = align_model.to(device)
    #     labels = processor.tokenizer.get_vocab()
    #     align_dictionary = {char.lower(): code for char,code in processor.tokenizer.get_vocab().items()}

    return align_dictionary
    

# adapted from https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py
def clean_whisper_transcript(transcript, model_dictionary=None, model_lang='en'):
    # Preprocess to split up and keep only words in dictionary
    for sdx, segment in enumerate(transcript):
        text = segment["text"].strip()

        # split into words
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = text

        clean_wrd = []
        for wdx, wrd in enumerate(per_word):
            if isinstance(model_dictionary, type(None)):
                clean_wrd.append(wrd)
            else:
                # filter for words containing model characters (done in whisperx pipeline so this allows for consistency)
                if any([c in model_dictionary.keys() for c in wrd]):
                    clean_wrd.append(wrd)
    return clean_wrd


def run_whisper(in_dir, out_dir, args=dict(),
                use_gpu=False, smokescreen=False, force_recompute=False, verbose=False):
    if verbose:
        print('starting run_whisper with')
        print(in_dir)
        print(out_dir)
    
    # Check if the provided directory path exists
    if not os.path.exists(in_dir):
        print(f"The input directory {in_dir} does not exist.")
        sys.exit(1)

    # Check if the provided path is a directory
    if not os.path.isdir(in_dir):
        print(f"The input path {in_dir} is not a directory.")
        sys.exit(1)

    # create a subdirectory to store all results
    # assert out_dir != in_dir
    out_dir = os.path.join(out_dir, 'whisper_out')
    os.makedirs(out_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(in_dir) if f.endswith('.wav')]

    if wav_files and not force_recompute:
        # skip processed .wav files
        tmp = list()
        for file in wav_files:
            output_name = os.path.splitext(file)[0] + ".csv"
            savepath = os.path.join(out_dir, output_name)
            if os.path.exists(savepath):
                print(f'Already produced output {savepath}. Skipping {file}.')
                continue
            tmp.append(file)
        wav_files = tmp
    if smokescreen: wav_files = wav_files[:1]
        
    if wav_files:
        device = "cuda:0" if use_gpu else "cpu"
        if verbose:
            print("\n\n====== Device type : {} ======".format(device))
        batch_size = 16 # reduce if low on GPU mem
        compute_type = "float16" if use_gpu else "int8"
        model_size = "large-v2" if not smokescreen else 'tiny.en'
        model = whisperx.load_model(model_size, device, compute_type=compute_type)
        transcribe_args = args['transcribe'] if 'transcribe' in args else dict()

        for file in wav_files:
            print("\n\nProcessing {}...".format(file))
            output_name = os.path.splitext(file)[0] + ".csv"
            savepath = os.path.join(out_dir, output_name)
            
            # define audio file
            filepath = os.path.join(in_dir, file)
            audio = whisperx.load_audio(filepath)
            if smokescreen:
                audio = audio[:len(audio) // 3]
            result = model.transcribe(audio, batch_size=batch_size, **transcribe_args)
            
            print(result)
            
            # 2. Get cleaned words for WER comparisons
            words = clean_whisper_transcript(result['segments'])
            print(words)
            words = [word.translate(str.maketrans('', '', string.punctuation)).upper() 
                     for word in words]
            print(words)

            with open(os.path.join(out_dir, 
                                   os.path.splitext(file)[0] + '.json'), 'w') as f:
                json.dump(result, f)
            df = pd.DataFrame({'Word': words})
            df.to_csv(savepath, index=False)
    
    
def run_whisperx(in_dir, out_dir, args=dict(),
                 use_gpu=False, smokescreen=False, force_recompute=False, verbose=False):
    # if smokescreen:
    #     from dask.distributed import print
    if verbose:
        print('starting run_whisperx with')
        print(in_dir)
        print(out_dir)
    
    # Check if the provided directory path exists
    if not os.path.exists(in_dir):
        print(f"The input directory {in_dir} does not exist.")
        sys.exit(1)

    # Check if the provided path is a directory
    if not os.path.isdir(in_dir):
        print(f"The input path {in_dir} is not a directory.")
        sys.exit(1)

    # create a subdirectory to store all results
    # assert out_dir != in_dir
    out_dir = os.path.join(out_dir, 'whisperx_out')
    os.makedirs(out_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(in_dir) if f.endswith('.wav')]

    if wav_files and not force_recompute:
        # skip processed .wav files
        tmp = list()
        for file in wav_files:
            output_name = os.path.splitext(file)[0] + ".csv"
            savepath = os.path.join(out_dir, output_name)
            if os.path.exists(savepath):
                print(f'Already produced output {savepath}. Skipping {file}.')
                continue
            tmp.append(file)
        wav_files = tmp
    if smokescreen: wav_files = wav_files[:1]
        
    if wav_files:
        device = "cuda:0" if use_gpu else "cpu"
        if verbose:
            print("\n\n====== Device type : {} ======".format(device))
        
        if args and 'whisper_dir' in args['whisperx_args']:
            whisper_dir = args['whisperx_args']['whisper_dir']
        else: whisper_dir = None
        
        if not whisper_dir:
            batch_size = 16 # reduce if low on GPU mem
            compute_type = "float16" if use_gpu else "int8"
            model_size = "large-v2" if not smokescreen else 'tiny.en'
            model = whisperx.load_model(model_size, device, compute_type=compute_type)
        model_a, metadata = whisperx.load_align_model(language_code="en", device=device)

        for file in wav_files:
            print("\n\nProcessing {}...".format(file))
            output_name = os.path.splitext(file)[0] + ".csv"
            savepath = os.path.join(out_dir, output_name)
            
            # define audio file
            filepath = os.path.join(in_dir, file)
            audio = whisperx.load_audio(filepath)
            if smokescreen:
                audio = audio[:len(audio) // 3]
            if whisper_dir:
                with open(os.path.join(whisper_dir, os.path.splitext(file)[0] + '.json'), 'r') as f:
                    result = json.load(f)
            else:
                transcribe_args = args['transcribe'] if 'transcribe' in args else dict()
                result = model.transcribe(audio, batch_size=batch_size, **transcribe_args)
            
            # 2. Align whisper output
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            
            words, onsets, offsets, confidence = [], [], [], []
            for segment in result["segments"]:
            
                word_list = segment['words']
                for element in word_list:
                    word = element['word']
                    onset_time = element['start']
                    offset_time = element['end']
                    prob = element['score']
                
                    newword = word.translate(str.maketrans('', '', string.punctuation)).upper()
                    words.append(newword)
                    onsets.append(int(onset_time * 1000))
                    offsets.append(int(offset_time * 1000))
                    confidence.append(prob)

            df = pd.DataFrame({'Word': words, 'Onset': onsets, 'Offset': offsets, 'Probability': confidence})
            df.to_csv(savepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated Speech Annotation using AI Models')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the input directory.')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='Flag to use GPU.')
    parser.add_argument('--whisper_only', action='store_true', default=False, 
                        help='Whether to run whisper (for speech transcription and phrase-level segmentation) instead of whisperX (for word-level segmentation)')
    parser.add_argument('--smokescreen', action='store_true', default=False, help='Flag for smokescreen runs for fast testing.')
    parser.add_argument('--force_recompute', action='store_true', default=False, help='Whether to force result recomputation.')
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"The directory {args.input_dir} does not exist.")
        sys.exit(1)

    if not os.path.isdir(args.input_dir):
        print(f"The path {args.input_dir} is not a directory.")
        sys.exit(1)

    if args.whisper_only:
        run_whisper(args.input_dir, args.output_dir, use_gpu=args.use_gpu, smokescreen=args.smokescreen, force_recompute=args.force_recompute)
    else:
        run_whisperx(args.input_dir, args.output_dir, use_gpu=args.use_gpu, smokescreen=args.smokescreen, force_recompute=args.force_recompute)
