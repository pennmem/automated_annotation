import os
import sys
import pandas as pd
import argparse
import numpy as np
from jiwer import wer
import eng_to_ipa as ipa
import re
import scipy.stats as st

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

# extract leading phoneme
def leading_phoneme(word):
    return ipa.convert(word)[0]


def word_error_rate(gt, sessnum, pred, verbose):
    # iterate through all csv files in the pred director
    wers = []

    # create lists to store return values
    onset_diffs = []
    correctly_annotated_words = []
    out_pred_onset = []
    out_gt_onset = []
    out_pred_probability = []

    for csvfile in os.listdir(pred):
        if csvfile.endswith(".csv"):
            if verbose:
                print("\n===== Processing model output for : {} =====".format(csvfile))

            df = pd.read_csv(os.path.join(pred, csvfile))

            # to handle edge cases
            if len(df) == 0:
                continue

            # convert to trial number
            trialNum = int(csvfile[:-4]) + 1

            # extract recalld word events
            rec_evs = gt[(gt["type"] == 'REC_WORD') & (gt["session"] == sessnum) & (gt["trial"] == trialNum)]
            #rec_evs = gt.query('type == "REC_WORD" and session == sessnum and trial == trialNum')
            gtwords = list(rec_evs["item_name"])
            gtOnsets = list(rec_evs["rectime"])


            '''
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
            '''

            correct = []
            pred_onsets = []

            # Words for which the nearest neighbor matched word pairs do not match should be analyzed separately
            matched_words = []
            matched_words_onsetdiff = []
            matched_words_probability = []
            matched_words_gtonset = []

            # mismatched words
            mismatched_words = []
            mismatched_words_onsetdiff = []
            mismatched_words_probability = []
            mismatched_words_gtonset = []

            for word, onset in zip(gtwords, gtOnsets):
                if word != 'VV':
                    minonset, idx = get_closest_time(df.Onset.astype(int), int(onset))
                    if abs(minonset - int(onset)) < DIFF_THRESHOLD:
                        correct.append(df.Word[idx])
                        pred_onsets.append(minonset)

                        # calculate the difference and append
                        onset_diffs.append(minonset - int(onset))

                        # add attributes to output
                        correctly_annotated_words.append(word)
                        out_gt_onset.append(int(onset))
                        out_pred_onset.append(minonset)
                        out_pred_probability.append(df.Probability[idx])

                        if word == df.Word[idx]:
                            matched_words.append(df.Word[idx]); matched_words_onsetdiff.append(minonset - int(onset))
                            matched_words_probability.append(df.Probability[idx]); matched_words_gtonset.append(int(onset))
                        else:
                            mismatched_words.append(df.Word[idx]); mismatched_words_onsetdiff.append(minonset - int(onset))
                            mismatched_words_probability.append(df.Probability[idx]); mismatched_words_gtonset.append(int(onset))

            matched_data = (matched_words, matched_words_onsetdiff, matched_words_probability, matched_words_gtonset)
            mismatched_data = (mismatched_words, mismatched_words_onsetdiff, mismatched_words_probability, mismatched_words_gtonset)
                    

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
                

    if verbose:
        print("\n\nAverage WER:  ", np.mean(wers))

    # return outputs
    # sanity check
    assert len(onset_diffs) == len(correctly_annotated_words) == len(out_pred_onset) == len(out_gt_onset) == len(out_pred_probability)
    return wers, np.array(onset_diffs), correctly_annotated_words, out_pred_onset, out_pred_probability, out_gt_onset, matched_data, mismatched_data



def run_phoneme_analysis(df, verbose=False):
    #fetch unique words
    unique_vocab = list(df['Word'].unique())
    phoneme_map = {}

    for vocab in unique_vocab:
        phoneme_map[vocab] = leading_phoneme(vocab)

    df['Phoneme'] = df['Word'].map(phoneme_map)
    result = df.groupby('Phoneme')['TimeDiff'].agg(['mean', 'std', 'count'])

    if verbose:
        print(result)
    return result

def run_recall_time_analysis(df, verbose=False):
    # 75 seconds recall time
    assert df['RecallTime'].max() <= 75000

    # 5000 ms step
    bins = list(range(0, 75001, 5000))

    # Create a new column 'RecallTime_bins' based on the bins
    df['RecallTime_bins'] = pd.cut(df['RecallTime'], bins, right=False)

    # Group by the bins and compute mean and std for 'TimeDiff'
    result = df.groupby('RecallTime_bins')['TimeDiff'].agg(['mean', 'std', 'count'])

    if verbose:
        print(result)
    return result


def run_confidence_analysis(df, verbose=False):
    # Create bins for the 'Probability' column
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    df['Probability_Bin'] = pd.cut(df['Probability'], bins=bins, labels=labels, include_lowest=True)

    # Group by 'Probability_Bin' and calculate the mean, standard deviation, and count for 'TimeDiff'
    result = df.groupby('Probability_Bin')['TimeDiff'].agg(['mean', 'std', 'count'])

    # Fill NaN values in the 'std' column with 0
    if verbose:
        print("\n\n")
        print(result)
    return result

def run_all_analysis(gt, pred, verbose=False, use_csv=False, csvpath=None):

    # load csv file if csv file is used
    if use_csv:
        if verbose:
            print("\nLoading GT data files...")
        gt_df = pd.read_csv(csvpath)

        if verbose:
            print("Done")


    # 1. word error rate analysis
    # recursively find how many patients there are
    sub_path = find_target_folder(pred)

    sublist = []
    seshlist = []
    werlist = []

    problem_sublist = []
    problem_seshlist = []
    problem_werlist = []

    # for good futures only
    gf_sublist = []
    gf_seshlist = []
    gf_werlist = []

    # other metrics
    gf_diff_means = []
    gf_diff_std = []

    # phoneme / confidence / time within recall phase etc
    aggr_words = []
    aggr_timediffs = []
    aggr_probs = []
    aggr_onsets = []

    # mismatched words
    mis_aggr_words = []
    mis_aggr_timediffs = []
    mis_aggr_probs = []
    mis_aggr_onsets = []


    # for each subject, compute the metrics separately.
    for ltpsub in os.listdir(sub_path):
        if os.path.isdir(os.path.join(sub_path, ltpsub)):
            print("Processing subject: {}....".format(ltpsub))
            sub_events = gt_df[gt_df["subject"] == ltpsub]

            # fetch all sessions
            single_sub_path = os.path.join(sub_path, ltpsub)
            sessions = [sesh for sesh in os.listdir(single_sub_path) if os.path.isdir(os.path.join(single_sub_path, sesh))]

            # perform analysis for each session.
            # then average across sessions

            sub_wers = []

            for sesh in sessions:
                pred_path = os.path.join(os.path.join(single_sub_path, sesh), 'whisperx_out')

                # skip if there are no csv files
                if len(os.listdir(pred_path)) == 0:
                    continue

                # extract session number
                match = re.search(r'session_(\d+)', sesh)
                sessnum = match.group(1)
                

                # word error rate for single session
                if verbose:
                    print("Processing session... ", sesh)
                wer, diff, word, pred_onset, pred_prob, gt_onset, match, mismatch = word_error_rate(sub_events, int(sessnum), pred_path, verbose)


                if (np.mean(wer) < 0.1):
                    sub_wers.append(np.mean(wer))

                    gf_sublist.append(ltpsub)
                    gf_seshlist.append(sesh)
                    gf_werlist.append(np.mean(wer))

                    # compute difference
                    gf_diff_means.append(np.mean(diff))
                    gf_diff_std.append(np.std(diff))

                    # aggregation for matched words
                    aggr_words.extend(match[0])
                    aggr_timediffs.extend(match[1])
                    aggr_probs.extend(match[2])
                    aggr_onsets.extend(match[3])

                    # aggregation for mismatched words
                    mis_aggr_words.extend(mismatch[0])
                    mis_aggr_timediffs.extend(mismatch[1])
                    mis_aggr_probs.extend(mismatch[2])
                    mis_aggr_onsets.extend(mismatch[3])

                else:
                    problem_sublist.append(ltpsub)
                    problem_seshlist.append(sesh)
                    problem_werlist.append(np.mean(wer))

                sublist.append(ltpsub)
                seshlist.append(sesh)
                werlist.append(np.mean(wer))

            if verbose and len(sub_wers) > 0:
                print("Subject mean: {}".format(np.mean(sub_wers)))



    # results.csv contains all session outputs, regardless of average WER
    analysis = {
        "subject" : sublist,
        "session" : seshlist,
        "wer" : [round(value, 4) for value in werlist]
    }
    out_df = pd.DataFrame(analysis)

    outpath = os.path.join(pred, "results.csv")
    out_df.to_csv(outpath)

    # problem_sessions only saves session outputs with WER > 0.1
    problem_sessions = {
        "subject" : problem_sublist,
        "session" : problem_seshlist,
        "wer" : [round(value, 4) for value in problem_werlist]
    }
    out_df2 = pd.DataFrame(problem_sessions)

    outpath = os.path.join(pred, "problem_sessions.csv")
    out_df2.to_csv(outpath)

    # good_futures.csv only saves session outputs with WER < 0.1
    good_futures = {
        "subject" : gf_sublist,
        "session" : gf_seshlist,
        "wer" : [round(value, 4) for value in gf_werlist],
        "diff_mean" : gf_diff_means,
        "diff_stdev" : gf_diff_std
    }
    out_df3 = pd.DataFrame(good_futures)

    outpath = os.path.join(pred, "good_futures.csv")
    out_df3.to_csv(outpath)

    # compute the per-subject average fetched from good_futures.csv
    result = out_df3.groupby('subject').agg({
    'wer': 'mean',
    'diff_mean': 'mean',
    'diff_stdev': 'mean'
    })

    # construct 95% confidence interval for each metric
    wer_interval = st.t.interval(confidence=0.95, df=len(result)-1, loc=result['wer'].mean(), scale=st.sem(result['wer']))
    mean_interval = st.t.interval(confidence=0.95, df=len(result)-1, loc=result['diff_mean'].mean(), scale=st.sem(result['diff_mean']))
    std_interval = st.t.interval(confidence=0.95, df=len(result)-1, loc=result['diff_stdev'].mean(), scale=st.sem(result['diff_stdev']))

    # Save the subject means to a separate csv file
    outpath = os.path.join(pred, "subject_means.csv")
    result.to_csv(outpath)

    # aggregated DataFrame for subsequent, word-level analysis
    data = {
    'Word': aggr_words,
    'TimeDiff' : aggr_timediffs,
    'Probability' : aggr_probs,
    'RecallTime' : aggr_onsets
    }

    mismatched_data = {
        'Word': mis_aggr_words,
        'TimeDiff' : mis_aggr_timediffs,
        'Probability' : mis_aggr_probs,
        'RecallTime' :  mis_aggr_onsets
    }


    print("\n\n====Performing subsequent analysis====")
    aggregate = pd.DataFrame(data)
    mismatch_aggregate = pd.DataFrame(mismatched_data)
    

    # 3. Bias/variability of automated methods broken out by
    # leading phonemes
    phon_results = run_phoneme_analysis(aggregate, verbose).sort_values(by='mean', ascending=False)
    phon_outtext = phon_results.to_string()

    #Word in word pool

    # Time within recall phase
    # recall duration is how long? (75s or 90s)?
    time_outtext = run_recall_time_analysis(aggregate, verbose).to_string()

    #Confidence level.
    conf_outtext = run_confidence_analysis(aggregate, verbose).to_string()

    # write summary statistics to a text file
    txtpath = os.path.join(pred, "summary_stats.txt")
    with open(txtpath, "w") as file:
        file.write("=====  Analysis Results  =====\n")
        file.write("mean WER : {:.4f}, CI: {}".format(result['wer'].mean(), wer_interval))
        file.write("\nmean onset difference (Prediction - GT) : {:.4f} ms, CI: {}".format(result['diff_mean'].mean(), mean_interval))
        file.write("\nstd onset difference (Prediction - GT) : {:.4f} ms, CI: {}\n\n".format(result['diff_stdev'].mean(), std_interval))

        file.write("\n\nMean Onset Difference for Matched Words: {}\n".format(aggregate['TimeDiff'].mean()))
        file.write("Mean Onset Difference for Mismatched Words: {}\n".format(mismatch_aggregate['TimeDiff'].mean()))

        file.write('\nLeading Phonemes analysis:\n')
        file.write(phon_outtext)
        file.write('\n\nRecall Time analysis:\n')
        file.write(time_outtext)
        file.write('\n\nConfidence level analysis:\n')
        file.write(conf_outtext)


    return



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

# helper function to recursively find the target directory
def find_target_folder(input_directory):
    # Walk through the directory recursively
    for dirpath, dirnames, filenames in os.walk(input_directory):
        # Check if current directory is "whisperx_out"
        if os.path.basename(dirpath) == "whisperx_out":
            # Check if all files in the directory are CSV files

            if all(filename.endswith('.csv') for filename in filenames):
                parent_directory = os.path.dirname(dirpath)
                # Count the number of folders in the parent directory
                session_directory = os.path.dirname(parent_directory)
                num_folders = sum([1 for name in os.listdir(session_directory) if os.path.isdir(os.path.join(session_directory, name))])
                #print(f"Parent folder name: {os.path.basename(session_directory)}")
                #print(f"Number of sessions in the patient: {num_folders}")

                subject_directory = os.path.dirname(session_directory)
                #print(subject_directory)
                num_subs = sum([1 for name in os.listdir(subject_directory) if os.path.isdir(os.path.join(subject_directory, name))])
                #print(f"Number of subjects: {num_subs}")

                # returns the directory where all subject folders (LTP-...) are located
                return subject_directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluates outputs of AI annotators based on different metrics')
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to the predictions (.csv) directory.')
    parser.add_argument('--gt_dir', type=str, help='Path to the ground truth data (in ann/par format).')
    parser.add_argument('--verbose', action='store_true', default=False, help='Using this flag will execute all optional print statements for ease with debugging.')
    parser.add_argument('--use_csv', action='store_true', default=False, help='Using this flag will make the program retrieve GT data from a single csv file.')
    parser.add_argument('--csvpath', type=str, default=None, help='Path to the ground truth data (in csv format).')
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

    if not args.use_csv and args.gt_dir == None:
        print(f"The directory {args.gt_dir} does not exist.")
        sys.exit(1)

    if not args.use_csv:
        if args.gt_dir != None and not os.path.isdir(args.gt_dir):
            print(f"The directory {args.gt_dir} does not exist.")
            sys.exit(1)

    if args.use_csv and args.csvpath == None:
        print(f"WARNING: Please specify path to the ground truth csv file.")
        sys.exit(1)

    # run the analysis
    run_all_analysis(args.gt_dir, args.pred_dir, args.verbose, args.use_csv, args.csvpath)


