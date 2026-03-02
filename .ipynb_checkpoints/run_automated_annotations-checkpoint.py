from automated_annot import run_whisperx
from cmldask import CMLDask
from dask.distributed import wait
import pickle
import os
import json
from datetime import datetime

#James' change
from automated_annot import run_whisperx, run_whisper


def current_time_string(fmt='%Y_%m_%d__%H_%M_%S'):
    return datetime.now().strftime(fmt)

def main(args):
    # create input/output dirs in Jupyter for convenience but run here
    # since Jupyter appears to have file permissions issues
    with open('input_dirs.pkl', 'rb') as f:
        all_input_dirs = pickle.load(f)
    with open('output_dirs.pkl', 'rb') as f:
        all_output_dirs = pickle.load(f)

    # aggregate run parameters
    tag = args.tag
    
    if 'whisperx' in tag.lower():
        func = run_whisperx
    elif 'whisper' in tag.lower():
        func = run_whisper
    else: raise ValueError()
    
    run_args = dict()
    run_args['meta'] = dict()
    run_args['meta']['run_start_timestamp'] = current_time_string('%Y-%m-%d_%H:%M:%S')
    run_args['cmd_args'] = vars(args)
    run_args['func_args'] = dict()
    run_args['func_args']['transcribe'] = dict()
    
    if 'long-prompt' in tag:
        # word pool needs to be computed separately for each session using WORD events
        run_args['func_args']['transcribe']['initial_prompt'] = 'The following is the word pool for a word list memory experiment. {wordpool}. Please transcribe the recorded audio of a subject recalling these words in any order along with other words they thought they studied. Be mindful to transcribe any homonyms with the words on the list as the actual words on the list'
    elif 'short-prompt' in tag:
        # word pool needs to be computed separately for each session using WORD events
        run_args['func_args']['transcribe']['initial_prompt'] = '{wordpool}'
    out = all_output_dirs[tag][0]
    run_dir = out.split(f'{tag}')
    # print(out)
    assert len(run_dir) == 2, f'Run tag {tag} occurs multiple times in output paths. Please select another run tag.'
    run_dir = os.path.join(run_dir[0], tag)
    with open(os.path.join(run_dir, 'run_params.json'), 'w') as f:
        json.dump(run_args, f)
    
    if args.smokescreen:  # for quick testing with reduced inputs
        # all_input_dirs[tag] = all_input_dirs[tag][:1]
        # all_output_dirs[tag] = all_output_dirs[tag][:1]
        
        '''
        # debug session from mismatched sessions issue
        for i, d in enumerate(all_input_dirs[tag]):
            if 'LTP123' in d and 'session_5' in d:
                print(i)
                break
        '''

        all_input_dirs[tag] = all_input_dirs[tag][i:i+1]
        all_output_dirs[tag] = all_output_dirs[tag][i:i+1]
        # put smokescreen outputs in test/ rather than in true results/
        dirs_temp = list()
        substring = 'results'
        for d in all_output_dirs[tag]:
            i = d.find(substring)
            d = d[:i + len(substring)].replace(substring, 'test/' + substring) + d[i + len(substring):]
            if not os.path.exists(d): os.makedirs(d)
            dirs_temp.append(d)
        all_output_dirs[tag] = dirs_temp
    else:
        # assert that input and output directories match on shared structure
        splits = ['train', 'val', 'test']
        for inp, out in zip(all_input_dirs[tag], all_output_dirs[tag]):
            assert inp != out
            out_split = None
            for split in splits:
                if split in out: out_split = split
            if isinstance(out_split, type(None)): raise ValueError
            out_strip = out.split(f'{tag}/{out_split}')[-1]
            fail_match = False
            if inp != out_strip:
                fail_match = True
            if fail_match:
                raise ValueError('Input/output directories do not match past {tag}/{split}!')
                
    # run models
    if args.use_dask:
        # if args.smokescreen: from dask.distributed import print
        dask_args = {'job_name': 'auto_annotate', 'memory_per_job': "9GB", 'max_n_jobs': 150,
                    'death_timeout': 600, 'extra': ['--no-dashboard'], 'log_directory': 'logs'}
        client = CMLDask.new_dask_client_slurm(**dask_args)
        dask_inputs = [all_input_dirs[tag], 
                       all_output_dirs[tag], 
                       [args.use_gpu] * len(all_output_dirs[tag]), 
                       [args.smokescreen] * len(all_output_dirs[tag]), 
                       [args.force_recompute] * len(all_output_dirs[tag])]
        futures = client.map(run_whisperx, *dask_inputs)
        wait(futures)
    else:
        for in_dir, out_dir in zip(all_input_dirs[tag], all_output_dirs[tag]):
            func(in_dir, out_dir, 
                 use_gpu=args.use_gpu, 
                 smokescreen=args.smokescreen, 
                 force_recompute=args.force_recompute)
    
    # save out run completion timestamp
    run_args['meta']['run_finished_timestamp'] = current_time_string('%Y-%m-%d_%H:%M:%S')
    with open(os.path.join(run_dir, 'run_params.json'), 'w') as f:
        json.dump(run_args, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Command-line arguments for script")
    parser.add_argument('--tag', type=str, default='base-whisperx', help="Tag value. Default is 'base-whisperx'.")
    parser.add_argument('--use-dask', action='store_true', help="Flag to use dask. Default is False.")
    parser.add_argument('--use-gpu', action='store_true', help="Flag to use GPU. Default is False.")
    parser.add_argument('--smokescreen', action='store_true', dest='smokescreen', 
                        help="Flag to enable smokescreen run for quick tests.")
    parser.add_argument('--force_recompute', action='store_true', 
                        help="Flag to force results recomputation. Otherwise, audio recordings with saved annotation outputs will be skipped.")
    args = parser.parse_args()
    main(args)
