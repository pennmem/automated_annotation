from automated_annot import run_whisperx
from cmldask import CMLDask
from dask.distributed import wait
import pickle
import os


def main(args):
    # create input/output dirs in Jupyter for convenience but run here
    # since Jupyter appears to have file permissions issues
    with open('input_dirs.pkl', 'rb') as f:
        all_input_dirs = pickle.load(f)
    with open('output_dirs.pkl', 'rb') as f:
        all_output_dirs = pickle.load(f)

    tag = args.tag

    if args.smokescreen:  # for quick testing with reduced inputs
        all_input_dirs[tag] = all_input_dirs[tag][:1]
        all_output_dirs[tag] = all_output_dirs[tag][:1]
        # put smokescreen outputs in test/ rather than in true results/
        dirs_temp = list()
        substring = 'results'
        for d in all_output_dirs[tag]:
            i = d.find(substring)
            d = d[:i + len(substring)].replace(substring, 'test/' + substring) + d[i + len(substring):]
            if not os.path.exists(d): os.makedirs(d)
            dirs_temp.append(d)
        all_output_dirs[tag] = dirs_temp
    
    if args.use_dask:
        if args.smokescreen: from dask.distributed import print
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
            run_whisperx(in_dir, out_dir, 
                         use_gpu=args.use_gpu, 
                         smokescreen=args.smokescreen, 
                         force_recompute=args.force_recompute)


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
