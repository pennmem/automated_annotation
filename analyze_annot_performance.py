import os
import sys
import pandas as pd
import argparse
import numpy as np
from jiwer import wer

DIFF_THRESHOLD = 1000

# imported from verify_annotation.py
def get_closest_time(testing_times, true_time):
    """
    returns: (best_diff, index_of_best_diff)
    """
    time_differences  = []
    diffs = [true_time-testing_time for testing_time in testing_times]
    abs_diffs = [abs(diff) for diff in diffs]
    min_diff = min(abs_diffs)
    min_index = abs_diffs.index(min_diff)
    return testing_times[min_index], min_index


def word_error_rate(gt, pred, verbose):
    # iterate through all csv files in the pred director
    wers = []

    for csvfile in os.listdir(pred):
        if csvfile.endswith(".csv"):
            if verbose:
                print("\n===== Processing model output for : {} =====".format(csvfile))

            df = pd.read_csv(os.path.join(pred, csvfile))

            # for each file, ensure that accompanying .lst and .ann exists
            lst = csvfile[:-4] + ".lst"
            ann = csvfile[:-4] + ".ann"
            par = csvfile[:-4] + ".par"

            if not os.path.exists(os.path.join(gt, lst)):
                print("The file {} does not exist in the specified directory.".format(lst))
                continue

            # load word list from .lst file
            wordlist = []
            with open(os.path.join(gt, lst), 'r') as f:
                for line in f:
                    if len(line) > 0:
                        wordlist.append(line.strip())

            if not os.path.exists(os.path.join(gt, ann)) and not os.path.exists(os.path.join(gt, par)):
                print("The files {} and {} does not exist in the specified directory.".format(ann, par))
                continue

            if not os.path.exists(os.path.join(gt, par)):

                # if there's only .ann file, convert it to par
                anntopar(gt, ann)

            # read a corresponding par file
            gtwords = []
            gtOnsets = []
            with open(os.path.join(gt, par), 'r') as f:
                for line in f:

                    # collect all recalls + intrusions
                    linearr = line.split()
                    #print(linearr)
                    #if linearr[2] != 'VV':
                    gtwords.append(linearr[2].strip())
                    gtOnsets.append(linearr[0].strip())

            correct = []
            pred_onsets = []
            for word, onset in zip(gtwords, gtOnsets):
                if word != 'VV':
                    minonset, idx = get_closest_time(df.Onset.astype(int), int(onset))
                    if abs(minonset - int(onset)) < DIFF_THRESHOLD:
                        correct.append(df.Word[idx])
                        pred_onsets.append(minonset)

                    

            filtered_gtwords = [word for word in gtwords if word != 'VV']
            indices = [i for i, word in enumerate(gtwords) if word != 'VV']
            filtered_gtonsets = [gtOnsets[i] for i in indices]
            #correct = [word for word in list(df.Word) if word in gtwords]

            hypothesis = ' '.join(correct)
            reference = ' '.join(filtered_gtwords)

            # edge case for empty refrence string
            if len(reference) == 0 and len(hypothesis) == 0:
                error = 0
                wers.append(error)
            elif len(reference) == 0:
                error = "NAN"
            else:
                error = wer(reference, hypothesis)
                wers.append(error)

            if verbose:
                print(filtered_gtwords)
                print(correct)
                print(filtered_gtonsets)
                print(pred_onsets)
                print(reference)
                print(hypothesis)
                print(error)
                
                


            # a proper word error rate cannot be calculated.
            # For now, word error rate will be calculated as percentage of 
            # correct recall that was correctly annoated. (excluding VV)
    if verbose:
        print("Average WER:  ", np.mean(wers))



def run_all_analysis(gt, pred, verbose=False):

    # 1. word error rate analysis
    word_error_rate(gt, pred, verbose)



# helper function to convert .ann files to .par to facilitate analysis
def anntopar(outdir, filename):
    parFile = filename[:-4] + ".par"
    annFile = open(os.path.join(outdir, filename),'r')
    parFile = open(os.path.join(outdir, parFile),'w')
    annLines = annFile.readlines()
    for annLine in annLines:
        if annLine[0] != '#':
            annLine = annLine.replace('\n','')
            annLine = annLine.split('\t')
            if annLine[0] != '':
                if annLine[2][0] == '<':
                    annLine[2] = 'VV'
                line = str(int(round(float(annLine[0])))) + '\t' + \
                       str(int(annLine[1])) + '\t' + annLine[2] + '\n'
                parFile.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluates outputs of AI annotators based on different metrics')
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to the predictions (.csv) directory.')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to the ground truth directory.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Using this flag will execute all optional print statements for debugging.')
    # Parse the arguments
    args = parser.parse_args()

    # Check if the provided directory path exists
    if not os.path.exists(args.pred_dir):
        print(f"The directory {args.pred_dir} does not exist.")
        sys.exit(1)

    # Check if the provided path is a directory
    if not os.path.isdir(args.pred_dir):
        print(f"The path {args.pred_dir} is not a directory.")
        sys.exit(1)

    if not os.path.exists(args.gt_dir):
        print(f"The directory {args.gt_dir} does not exist.")
        sys.exit(1)
    if not os.path.isdir(args.gt_dir):
        print(f"The path {args.gt_dir} is not a directory.")
        sys.exit(1)

    # run the analysis
    run_all_analysis(args.gt_dir, args.pred_dir, args.verbose)


