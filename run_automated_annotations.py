from automated_annot import run_whisperx
from cmldask import CMLDask
from dask.distributed import wait
import pickle


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
    
    if args.use_dask:
        if args.smokescreen: from dask.distributed import print
        dask_args = {'job_name': 'auto_annotate', 'memory_per_job': "9GB", 'max_n_jobs': 35,
                    'death_timeout': 600, 'extra': ['--no-dashboard'], 'log_directory': 'logs'}
        client = CMLDask.new_dask_client_slurm(**dask_args)
        dask_inputs = [all_input_dirs[tag], all_output_dirs[tag]]
        futures = client.map(run_whisperx, *dask_inputs)
        wait(futures)
    else:
        for in_dir, out_dir in zip(all_input_dirs[tag], all_output_dirs[tag]):
            run_whisperx(in_dir, out_dir, use_gpu=args.use_gpu, smokescreen=args.smokescreen)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Command-line arguments for script")
    parser.add_argument('--tag', type=str, default='base-whisperx', help="Tag value. Default is 'base-whisperx'.")
    parser.add_argument('--use-dask', action='store_true', help="Flag to use dask. Default is False.")
    parser.add_argument('--use-gpu', action='store_true', help="Flag to use GPU. Default is False.")
    parser.add_argument('--smokescreen', action='store_true', dest='smokescreen', help="Flag to disable smokescreen. Default is True.")
    args = parser.parse_args()
    main(args)
