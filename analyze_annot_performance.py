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


def build_gt_events_csv(experiment='ltpFR2', output_path=None):
    """Build a ground-truth events CSV from CML for all subjects/sessions.

    Loads the CML data index, iterates over every session for the given
    experiment, reads events via CMLReader, filters to WORD and REC_WORD
    types, and concatenates into a single DataFrame saved to output_path.

    Returns the concatenated DataFrame.
    """
    from cmlreaders import CMLReader, get_data_index

    df = get_data_index()
    exp_df = df[df['experiment'] == experiment]

    all_frames = []
    for _, row in exp_df.iterrows():
        subject = row['subject']
        session = row['session']
        montage = row.get('montage', 0)
        localization = row.get('localization', 0)
        try:
            reader = CMLReader(subject, experiment, session,
                               montage=montage, localization=localization)
            evs = reader.load('events')
            filtered = evs.query("type == 'WORD' or type == 'REC_WORD'")
            if len(filtered) > 0:
                all_frames.append(filtered)
        except Exception as e:
            print(f"WARNING: Failed to load events for {subject} session {session}: {e}")
            continue

    if not all_frames:
        raise RuntimeError(f"No events loaded for experiment {experiment}")

    result = pd.concat(all_frames, ignore_index=True)
    print(f"Built GT events: {len(result)} rows, "
          f"{result['subject'].nunique()} subjects, "
          f"{experiment}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    return result


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
    out_trial_nums = []

    # per-trial data for ROC analysis
    trial_roc_data = []

    for csvfile in os.listdir(pred):
        if csvfile.endswith(".csv"):
            if verbose:
                print("\n===== Processing model output for : {} =====".format(csvfile))

            df = pd.read_csv(os.path.join(pred, csvfile))

            # to handle edge cases
            df = df.dropna(subset=['Onset', 'Offset']).reset_index(drop=True)
            if len(df) == 0:
                continue

            # convert to trial number
            trialNum = int(csvfile[:-4]) + 1

            # extract recalld word events
            rec_evs = gt[(gt["type"] == 'REC_WORD') & (gt["session"] == sessnum) & (gt["trial"] == trialNum)]
            #rec_evs = gt.query('type == "REC_WORD" and session == sessnum and trial == trialNum')
            gtwords = list(rec_evs["item_name"])
            gtOnsets = list(rec_evs["rectime"])

            # collect per-trial data for ROC
            gt_word_set = set(w.upper() for w in gtwords if w != 'VV')
            pred_word_list = list(df['Word'].astype(str).str.upper())
            pred_prob_list = list(df['Probability'])
            trial_roc_data.append({
                'gt_words': gt_word_set,
                'pred_words': pred_word_list,
                'pred_probs': pred_prob_list,
                'trial': trialNum,
            })


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
            matched_words_predonset = []
            matched_words_trial = []

            # mismatched words
            mismatched_words = []
            mismatched_words_onsetdiff = []
            mismatched_words_probability = []
            mismatched_words_gtonset = []

            for word, onset in zip(gtwords, gtOnsets):
                if word != 'VV':
                    minonset, idx = get_closest_time(df.Onset.astype(int), int(onset))
                    if abs(minonset - int(onset)) < DIFF_THRESHOLD:
                        correct.append(str(df.Word[idx]))
                        pred_onsets.append(minonset)

                        # calculate the difference and append
                        onset_diffs.append(minonset - int(onset))

                        # add attributes to output
                        correctly_annotated_words.append(word)
                        out_gt_onset.append(int(onset))
                        out_pred_onset.append(minonset)
                        out_pred_probability.append(df.Probability[idx])
                        out_trial_nums.append(trialNum)

                        if word == df.Word[idx]:
                            matched_words.append(df.Word[idx]); matched_words_onsetdiff.append(minonset - int(onset))
                            matched_words_probability.append(df.Probability[idx]); matched_words_gtonset.append(int(onset))
                            matched_words_predonset.append(minonset); matched_words_trial.append(trialNum)
                        else:
                            mismatched_words.append(df.Word[idx]); mismatched_words_onsetdiff.append(minonset - int(onset))
                            mismatched_words_probability.append(df.Probability[idx]); mismatched_words_gtonset.append(int(onset))

            matched_data = (matched_words, matched_words_onsetdiff, matched_words_probability, matched_words_gtonset, matched_words_predonset, matched_words_trial)
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
    assert len(onset_diffs) == len(correctly_annotated_words) == len(out_pred_onset) == len(out_gt_onset) == len(out_pred_probability) == len(out_trial_nums)
    return wers, np.array(onset_diffs), correctly_annotated_words, out_pred_onset, out_pred_probability, out_gt_onset, matched_data, mismatched_data, out_trial_nums, trial_roc_data



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
    # Drop rows with missing RecallTime
    n_before = len(df)
    df = df.dropna(subset=['RecallTime']).copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Warning: dropped {n_dropped}/{n_before} rows with NaN RecallTime")

    if len(df) == 0:
        print("Warning: no valid RecallTime values, skipping recall time analysis")
        return pd.DataFrame(columns=['mean', 'std', 'count'])

    # Determine recall duration from data, rounded up to nearest 5s
    max_recall = df['RecallTime'].max()
    recall_duration = int(np.ceil(max_recall / 5000) * 5000)

    # 5000 ms step
    bins = list(range(0, recall_duration + 1, 5000))

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

### BROKEN NEED TO CHECK
def run_regression_analysis(df, verbose=False):
    """Linear regression between predicted and manual (GT) onsets.

    Computes Y(i) = β0 + β1·X(i) + e(i) where X = predicted onset,
    Y = manual onset, following the framework from the reference paper.
    Returns a dict with slope, intercept, r_squared, residuals, and
    absolute deviation (AD) = mean |residual|.
    """
    gt_onset   = df['RecallTime'].values
    pred_onset = df['PredOnset'].values if 'PredOnset' in df.columns else (df['RecallTime'] + df['TimeDiff']).values

    # drop any NaN pairs
    mask = ~(np.isnan(gt_onset) | np.isnan(pred_onset))
    gt_onset   = gt_onset[mask]
    pred_onset = pred_onset[mask]

    if len(gt_onset) < 3:
        if verbose:
            print("Warning: not enough data points for regression")
        return {'slope': np.nan, 'intercept': np.nan, 'r_squared': np.nan,
                'residuals': np.array([]), 'ad': np.nan, 'n': 0}

    slope, intercept, r_value, p_value, std_err = st.linregress(pred_onset, gt_onset)
    r_squared = r_value ** 2
    predicted_manual = intercept + slope * pred_onset
    residuals = predicted_manual - gt_onset
    ad = np.mean(np.abs(residuals))

    if verbose:
        print(f"Regression: slope={slope:.4f}, intercept={intercept:.2f}, "
              f"R²={r_squared:.4f}, AD={ad:.2f} ms, n={len(gt_onset)}")

    return {
        'slope':     slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'residuals': residuals,
        'ad':        ad,
        'n':         len(gt_onset),
    }

### BROKEN NEED TO CHECK
def compute_regression_residuals(df):
    """Add a 'RegressionResidual' column to df.

    Fits Y(i) = β0 + β1·X(i) where X = predicted onset, Y = manual onset,
    then stores ê(i) = ŷ(i) − Y(i) as the regression residual for each row.
    Unlike raw TimeDiff, residuals remove systematic measurement bias.
    """
    gt_onset   = df['RecallTime'].values.astype(float)
    pred_onset = df['PredOnset'].values.astype(float) if 'PredOnset' in df.columns else (df['RecallTime'] + df['TimeDiff']).values.astype(float)

    mask = ~(np.isnan(gt_onset) | np.isnan(pred_onset))
    if mask.sum() < 3:
        df['RegressionResidual'] = np.nan
        return df

    slope, intercept, *_ = st.linregress(pred_onset[mask], gt_onset[mask])
    predicted_manual = intercept + slope * pred_onset
    df['RegressionResidual'] = predicted_manual - gt_onset
    return df

### BROKEN NEED TO CHECK
def run_roc_analysis(roc_trials, thresholds=None, verbose=False):
    """Compute ROC curve for word identification accuracy.

    For each confidence threshold, computes:
    - Hit rate: proportion of GT words found in the (filtered) predicted set
    - False alarm rate: proportion of predicted words NOT in the GT set

    Parameters
    ----------
    roc_trials : list of dict
        Each dict has 'gt_words' (set), 'pred_words' (list), 'pred_probs' (list),
        'subject', 'session', 'trial'.
    thresholds : array-like, optional
        Confidence thresholds to sweep. Default: 0.0 to 1.0 in 0.1 steps.

    Returns
    -------
    roc_df : DataFrame with columns Threshold, HitRate, FalseAlarmRate, HitRate_SE, FalseAlarmRate_SE
    auc : float, area under the ROC curve
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.1)

    roc_rows = []
    for thresh in thresholds:
        trial_hits = []
        trial_fas = []

        for trial in roc_trials:
            gt_words = trial['gt_words']
            # filter predicted words by confidence threshold
            filtered_pred = [w for w, p in zip(trial['pred_words'], trial['pred_probs'])
                             if p >= thresh]
            filtered_set = set(filtered_pred)

            # hit rate: what fraction of GT words appear in filtered predictions?
            if len(gt_words) > 0:
                hits = sum(1 for w in gt_words if w in filtered_set)
                trial_hits.append(hits / len(gt_words))

            # false alarm rate: what fraction of predicted words are NOT in GT?
            if len(filtered_pred) > 0:
                fas = sum(1 for w in filtered_pred if w not in gt_words)
                trial_fas.append(fas / len(filtered_pred))
            elif len(gt_words) > 0:
                trial_fas.append(0.0)

        hit_rate = np.mean(trial_hits) if trial_hits else 0.0
        fa_rate = np.mean(trial_fas) if trial_fas else 0.0
        hit_se = np.std(trial_hits) / np.sqrt(len(trial_hits)) if len(trial_hits) > 1 else 0.0
        fa_se = np.std(trial_fas) / np.sqrt(len(trial_fas)) if len(trial_fas) > 1 else 0.0

        roc_rows.append({
            'Threshold': round(thresh, 2),
            'HitRate': hit_rate,
            'FalseAlarmRate': fa_rate,
            'HitRate_SE': hit_se,
            'FalseAlarmRate_SE': fa_se,
        })

    roc_df = pd.DataFrame(roc_rows)

    # AUC via trapezoidal integration (sorted by ascending FA rate)
    sorted_df = roc_df.sort_values('FalseAlarmRate')
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    auc = _trapz(sorted_df['HitRate'].values, sorted_df['FalseAlarmRate'].values)

    if verbose:
        print(f"ROC AUC: {auc:.3f}")
        print(roc_df.to_string(index=False))

    return roc_df, auc

### BROKEN NEED TO CHECK
def run_onset_correlation_analysis(df, verbose=False):
    """Correlate predicted vs manual onset times at three levels.

    Following the approach from the reference paper:
    1. Within-list: correlation per (subject, session, trial)
    2. Within-subject: correlation per subject (aggregated across trials)
    3. Overall: single correlation across all data

    Returns a dict with 'within_list', 'within_subject', 'overall' DataFrames/values.
    """
    df = df.dropna(subset=['RecallTime', 'PredOnset']).copy()

    results = {}

    # 1. Within-list correlations
    list_corrs = []
    for (subj, sess, trial), grp in df.groupby(['Subject', 'Session', 'Trial']):
        if len(grp) >= 3:
            r, p = st.pearsonr(grp['PredOnset'], grp['RecallTime'])
            list_corrs.append({
                'Subject': subj, 'Session': sess, 'Trial': trial,
                'r': r, 'p': p, 'n': len(grp),
            })
    results['within_list'] = pd.DataFrame(list_corrs)

    # 2. Within-subject correlations
    subj_corrs = []
    for subj, grp in df.groupby('Subject'):
        if len(grp) >= 3:
            r, p = st.pearsonr(grp['PredOnset'], grp['RecallTime'])
            subj_corrs.append({
                'Subject': subj, 'r': r, 'p': p, 'n': len(grp),
            })
    results['within_subject'] = pd.DataFrame(subj_corrs)

    # 3. Overall correlation
    if len(df) >= 3:
        r, p = st.pearsonr(df['PredOnset'], df['RecallTime'])
        results['overall'] = {'r': r, 'p': p, 'n': len(df)}
    else:
        results['overall'] = {'r': np.nan, 'p': np.nan, 'n': len(df)}

    if verbose:
        wl = results['within_list']
        ws = results['within_subject']
        ov = results['overall']
        if len(wl) > 0:
            print(f"Within-list correlations: mean r={wl['r'].mean():.4f}, min r={wl['r'].min():.4f}, n_lists={len(wl)}")
        if len(ws) > 0:
            print(f"Within-subject correlations: mean r={ws['r'].mean():.4f}, min r={ws['r'].min():.4f}, n_subjects={len(ws)}")
        print(f"Overall correlation: r={ov['r']:.4f}, p={ov['p']:.2e}, n={ov['n']}")

    return results


def run_word_analysis(df, verbose=False):
    """Onset error grouped by word within the wordpool."""
    result = df.groupby('Word')['TimeDiff'].agg(['mean', 'std', 'count']).sort_values(by='mean', ascending=True)
    if verbose:
        print(result)
    return result


def run_cluster_analysis(df, cluster_threshold_ms=1500, verbose=False):
    """Compare onset error for standalone vs clustered words.

    A word is 'clustered' if the previous word's ground-truth onset is
    within cluster_threshold_ms of the current word's onset.
    Words are sorted by RecallTime within the aggregate, so consecutive
    rows approximate consecutive recalls within a trial.
    """
    df_sorted = df.sort_values('RecallTime').copy()
    prev_onset = df_sorted['RecallTime'].shift(1)
    ioi = df_sorted['RecallTime'] - prev_onset
    df_sorted['WordType'] = np.where(ioi <= cluster_threshold_ms, 'Clustered', 'Standalone')
    # First word in each trial has NaN IOI → standalone
    df_sorted['WordType'] = df_sorted['WordType'].fillna('Standalone')

    result = df_sorted.groupby('WordType')['TimeDiff'].agg(['mean', 'std', 'count'])
    if verbose:
        print(result)
    return result, df_sorted

def run_all_analysis(gt, pred, verbose=False, use_csv=False, csvpath=None,
                     output_subdir='whisperx_out', experiment='ltpFR2'):

    # load csv file if csv file is used
    if use_csv:
        if csvpath and os.path.exists(csvpath):
            if verbose:
                print(f"\nLoading GT data from {csvpath}...")
            gt_df = pd.read_csv(csvpath)
        else:
            print(f"GT CSV not found at {csvpath}, building from CML...")
            gt_df = build_gt_events_csv(experiment=experiment, output_path=csvpath)

        if verbose:
            print(f"Done. GT has {gt_df['subject'].nunique()} subjects, {len(gt_df)} rows.")

    print(gt_df)

    # 1. word error rate analysis
    # recursively find all subject directories across all splits
    subject_entries = find_target_folder(pred, output_subdir=output_subdir)
    if verbose:
        print(f"Found {len(subject_entries)} subject directories across all splits")

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
    aggr_subjects = []
    aggr_sessions = []
    aggr_trials = []
    aggr_pred_onsets = []

    # mismatched words
    mis_aggr_words = []
    mis_aggr_timediffs = []
    mis_aggr_probs = []
    mis_aggr_onsets = []

    # ROC: per-trial word-level comparisons
    roc_trials = []  # list of dicts: {gt_words, pred_words, pred_probs, subject, session, trial}


    # for each subject, compute the metrics separately.
    for single_sub_path, ltpsub in subject_entries:
            print("Processing subject: {}....".format(ltpsub))
            sub_events = gt_df[gt_df["subject"] == ltpsub]

            # fetch all sessions
            sessions = [sesh for sesh in os.listdir(single_sub_path) if os.path.isdir(os.path.join(single_sub_path, sesh))]

            # perform analysis for each session.
            # then average across sessions

            sub_wers = []

            for sesh in sessions:
                pred_path = os.path.join(single_sub_path, sesh, output_subdir)

                # skip if there are no csv files
                if len(os.listdir(pred_path)) == 0:
                    continue

                # extract session number
                match = re.search(r'session_(\d+)', sesh)
                sessnum = match.group(1)
                

                # word error rate for single session
                if verbose:
                    print("Processing session... ", sesh)
                wer, diff, word, pred_onset, pred_prob, gt_onset, match, mismatch, trial_nums, trial_roc = word_error_rate(sub_events, int(sessnum), pred_path, verbose)


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
                    aggr_pred_onsets.extend(match[4])
                    aggr_trials.extend(match[5])
                    n_matched = len(match[0])
                    aggr_subjects.extend([ltpsub] * n_matched)
                    aggr_sessions.extend([sesh] * n_matched)

                    # aggregation for mismatched words
                    mis_aggr_words.extend(mismatch[0])
                    mis_aggr_timediffs.extend(mismatch[1])
                    mis_aggr_probs.extend(mismatch[2])
                    mis_aggr_onsets.extend(mismatch[3])

                    # ROC trial data
                    for trd in trial_roc:
                        trd['subject'] = ltpsub
                        trd['session'] = sesh
                    roc_trials.extend(trial_roc)

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
    'RecallTime' : aggr_onsets,
    'PredOnset' : aggr_pred_onsets,
    'Subject' : aggr_subjects,
    'Session' : aggr_sessions,
    'Trial' : aggr_trials,
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

    # Add regression residuals (bias-corrected onset error)
    aggregate = compute_regression_residuals(aggregate)
    mismatch_aggregate = compute_regression_residuals(mismatch_aggregate)

    # Save aggregate word-level data for downstream plotting
    aggregate.to_csv(os.path.join(pred, "aggregate_words.csv"), index=False)
    mismatch_aggregate.to_csv(os.path.join(pred, "aggregate_mismatched.csv"), index=False)

    # 3. Bias/variability of automated methods broken out by
    # leading phonemes
    phon_results = run_phoneme_analysis(aggregate, verbose).sort_values(by='mean', ascending=False)
    phon_outtext = phon_results.to_string()

    #Word in word pool
    word_results = run_word_analysis(aggregate, verbose)
    word_outtext = word_results.to_string()

    # Standalone vs clustered
    cluster_results, cluster_df = run_cluster_analysis(aggregate, verbose=verbose)
    cluster_outtext = cluster_results.to_string()

    # Time within recall phase
    # recall duration is how long? (75s or 90s)?
    time_results = run_recall_time_analysis(aggregate, verbose)
    time_outtext = time_results.to_string()

    #Confidence level.
    conf_results = run_confidence_analysis(aggregate, verbose)
    conf_outtext = conf_results.to_string()

    # Regression R² (pred onset vs manual onset)
    regression_results = run_regression_analysis(aggregate, verbose)

    # ROC analysis for word identification accuracy
    roc_df, roc_auc = run_roc_analysis(roc_trials, verbose=verbose)
    roc_df.to_csv(os.path.join(pred, "roc_curve.csv"), index=False)

    # Onset correlation analysis (within-list, within-subject, overall)
    correlation_results = run_onset_correlation_analysis(aggregate, verbose=verbose)
    if len(correlation_results['within_list']) > 0:
        correlation_results['within_list'].to_csv(os.path.join(pred, "correlations_within_list.csv"), index=False)
    if len(correlation_results['within_subject']) > 0:
        correlation_results['within_subject'].to_csv(os.path.join(pred, "correlations_within_subject.csv"), index=False)

    # compute dataset counts
    n_total_subjects  = out_df['subject'].nunique()
    n_total_sessions  = len(out_df)
    n_good_subjects   = out_df3['subject'].nunique()
    n_good_sessions   = len(out_df3)
    n_problem_sessions = len(out_df2)
    n_matched_words   = len(aggregate)
    n_mismatched_words = len(mismatch_aggregate)
    n_roc_trials      = len(roc_trials)
    n_total_recordings = n_matched_words + n_mismatched_words

    counts = {
        'n_total_subjects':   n_total_subjects,
        'n_total_sessions':   n_total_sessions,
        'n_good_subjects':    n_good_subjects,
        'n_good_sessions':    n_good_sessions,
        'n_problem_sessions': n_problem_sessions,
        'n_matched_words':    n_matched_words,
        'n_mismatched_words': n_mismatched_words,
        'n_total_recordings': n_total_recordings,
        'n_roc_trials':       n_roc_trials,
    }

    print(f'\n{"="*60}')
    print(f'  Dataset Counts')
    print(f'{"="*60}')
    print(f'  Total subjects:       {n_total_subjects}')
    print(f'  Total sessions:       {n_total_sessions}')
    print(f'  Good sessions (WER<0.1): {n_good_sessions} ({n_good_subjects} subjects)')
    print(f'  Problem sessions:     {n_problem_sessions}')
    print(f'  Matched words:        {n_matched_words}')
    print(f'  Mismatched words:     {n_mismatched_words}')
    print(f'  Total word pairs:     {n_total_recordings}')
    print(f'  ROC trials:           {n_roc_trials}')
    print(f'{"="*60}\n')

    # write summary statistics to a text file
    txtpath = os.path.join(pred, "summary_stats.txt")
    with open(txtpath, "w") as file:
        file.write("=====  Analysis Results  =====\n")
        file.write(f"Total subjects: {n_total_subjects}\n")
        file.write(f"Total sessions: {n_total_sessions}\n")
        file.write(f"Good sessions (WER<0.1): {n_good_sessions} ({n_good_subjects} subjects)\n")
        file.write(f"Problem sessions: {n_problem_sessions}\n")
        file.write(f"Matched word pairs: {n_matched_words}\n")
        file.write(f"Mismatched word pairs: {n_mismatched_words}\n")
        file.write(f"ROC trials: {n_roc_trials}\n\n")
        file.write("mean WER : {:.4f}, CI: {}".format(result['wer'].mean(), wer_interval))
        file.write("\nmean onset difference (Prediction - GT) : {:.4f} ms, CI: {}".format(result['diff_mean'].mean(), mean_interval))
        file.write("\nstd onset difference (Prediction - GT) : {:.4f} ms, CI: {}\n\n".format(result['diff_stdev'].mean(), std_interval))

        file.write("\n\nRegression (pred vs manual onset): R²={:.4f}, slope={:.4f}, intercept={:.2f}, AD={:.2f} ms, n={}\n".format(
            regression_results['r_squared'], regression_results['slope'],
            regression_results['intercept'], regression_results['ad'], regression_results['n']))

        file.write("\nMean Onset Difference for Matched Words: {}\n".format(aggregate['TimeDiff'].mean()))
        file.write("Mean Onset Difference for Mismatched Words: {}\n".format(mismatch_aggregate['TimeDiff'].mean()))

        file.write('\nLeading Phonemes analysis:\n')
        file.write(phon_outtext)
        file.write('\n\nWord-level analysis:\n')
        file.write(word_outtext)
        file.write('\n\nStandalone vs Clustered analysis:\n')
        file.write(cluster_outtext)
        file.write('\n\nRecall Time analysis:\n')
        file.write(time_outtext)
        file.write('\n\nConfidence level analysis:\n')
        file.write(conf_outtext)

        file.write(f'\n\nROC Analysis (word identification):\n')
        file.write(f'AUC: {roc_auc:.3f}\n')
        file.write(roc_df.to_string(index=False))

        file.write(f'\n\nOnset Correlation Analysis:\n')
        ov = correlation_results['overall']
        file.write(f'Overall: r={ov["r"]:.4f}, p={ov["p"]:.2e}, n={ov["n"]}\n')
        ws = correlation_results['within_subject']
        if len(ws) > 0:
            file.write(f'Within-subject: mean r={ws["r"].mean():.4f}, min r={ws["r"].min():.4f}, n_subjects={len(ws)}\n')
        wl = correlation_results['within_list']
        if len(wl) > 0:
            file.write(f'Within-list: mean r={wl["r"].mean():.4f}, min r={wl["r"].min():.4f}, n_lists={len(wl)}\n')

    return {
        # word-level aggregates
        'aggregate':          aggregate,
        'mismatch_aggregate': mismatch_aggregate,
        # session / subject summaries
        'results':            out_df,
        'good_futures':       out_df3,
        'problem_sessions':   out_df2,
        'subject_means':      result,
        # confidence intervals
        'wer_ci':             wer_interval,
        'mean_ci':            mean_interval,
        'std_ci':             std_interval,
        # sub-analyses (DataFrames)
        'phoneme':            phon_results,
        'word':               word_results,
        'cluster':            cluster_results,
        'cluster_df':         cluster_df,
        'recall_time':        time_results,
        'confidence':         conf_results,
        'regression':         regression_results,
        # ROC analysis
        'roc':                roc_df,
        'roc_auc':            roc_auc,
        # onset correlations
        'correlations':       correlation_results,
        # dataset counts
        'counts':             counts,
    }


def load_analysis(pred_dir, verbose=False):
    """Load previously saved analysis results and re-derive sub-analyses.

    Reads the CSVs written by run_all_analysis() and re-computes phoneme,
    word, cluster, recall-time, confidence, regression, ROC, and correlation
    analyses from the saved aggregate data. This avoids re-running the
    expensive WER computation.

    Parameters
    ----------
    pred_dir : str
        Path to the results directory (same as `pred` passed to run_all_analysis).
    verbose : bool
        Print progress info.

    Returns
    -------
    dict with the same keys as run_all_analysis().
    """
    # Load saved CSVs
    results_df    = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0)
    good_futures  = pd.read_csv(os.path.join(pred_dir, 'good_futures.csv'), index_col=0)
    problem_df    = pd.read_csv(os.path.join(pred_dir, 'problem_sessions.csv'), index_col=0)
    subject_means = pd.read_csv(os.path.join(pred_dir, 'subject_means.csv'), index_col=0)
    aggregate     = pd.read_csv(os.path.join(pred_dir, 'aggregate_words.csv'))
    mismatch_agg  = pd.read_csv(os.path.join(pred_dir, 'aggregate_mismatched.csv'))
    roc_df        = pd.read_csv(os.path.join(pred_dir, 'roc_curve.csv'))

    corr_wl_path = os.path.join(pred_dir, 'correlations_within_list.csv')
    corr_ws_path = os.path.join(pred_dir, 'correlations_within_subject.csv')

    if verbose:
        print(f'Loaded from {pred_dir}:')
        print(f'  {len(results_df)} total sessions, {len(good_futures)} good, {len(problem_df)} problem')
        print(f'  {len(aggregate)} matched words, {len(mismatch_agg)} mismatched')

    # Confidence intervals
    n = len(subject_means)
    wer_ci  = st.t.interval(confidence=0.95, df=n-1, loc=subject_means['wer'].mean(),
                            scale=st.sem(subject_means['wer']))
    mean_ci = st.t.interval(confidence=0.95, df=n-1, loc=subject_means['diff_mean'].mean(),
                            scale=st.sem(subject_means['diff_mean']))
    std_ci  = st.t.interval(confidence=0.95, df=n-1, loc=subject_means['diff_stdev'].mean(),
                            scale=st.sem(subject_means['diff_stdev']))

    # Re-derive sub-analyses from aggregate
    phon_results    = run_phoneme_analysis(aggregate.copy(), verbose)
    word_results    = run_word_analysis(aggregate.copy(), verbose)
    cluster_results, cluster_df = run_cluster_analysis(aggregate.copy(), verbose=verbose)
    time_results    = run_recall_time_analysis(aggregate.copy(), verbose)
    conf_results    = run_confidence_analysis(aggregate.copy(), verbose)
    regression      = run_regression_analysis(aggregate.copy(), verbose)

    # ROC AUC from saved curve
    sorted_roc = roc_df.sort_values('FalseAlarmRate')
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    roc_auc = _trapz(sorted_roc['HitRate'].values, sorted_roc['FalseAlarmRate'].values)

    # Correlations: load from CSV if available, otherwise re-derive
    correlations = {}
    if os.path.exists(corr_wl_path):
        correlations['within_list'] = pd.read_csv(corr_wl_path)
    else:
        correlations['within_list'] = pd.DataFrame()
    if os.path.exists(corr_ws_path):
        correlations['within_subject'] = pd.read_csv(corr_ws_path)
    else:
        correlations['within_subject'] = pd.DataFrame()

    # Overall correlation
    agg_clean = aggregate.dropna(subset=['RecallTime', 'PredOnset'])
    if len(agg_clean) >= 3:
        r, p = st.pearsonr(agg_clean['PredOnset'], agg_clean['RecallTime'])
        correlations['overall'] = {'r': r, 'p': p, 'n': len(agg_clean)}
    else:
        correlations['overall'] = {'r': np.nan, 'p': np.nan, 'n': len(agg_clean)}

    # Dataset counts
    counts = {
        'n_total_subjects':   results_df['subject'].nunique(),
        'n_total_sessions':   len(results_df),
        'n_good_subjects':    good_futures['subject'].nunique(),
        'n_good_sessions':    len(good_futures),
        'n_problem_sessions': len(problem_df),
        'n_matched_words':    len(aggregate),
        'n_mismatched_words': len(mismatch_agg),
        'n_total_recordings': len(aggregate) + len(mismatch_agg),
        'n_roc_trials':       0,  # not recoverable from saved data
    }

    if verbose:
        print(f'  Re-derived: phoneme({len(phon_results)}), word({len(word_results)}), '
              f'regression(R²={regression["r_squared"]:.4f}), ROC AUC={roc_auc:.3f}')

    return {
        'aggregate':          aggregate,
        'mismatch_aggregate': mismatch_agg,
        'results':            results_df,
        'good_futures':       good_futures,
        'problem_sessions':   problem_df,
        'subject_means':      subject_means,
        'wer_ci':             wer_ci,
        'mean_ci':            mean_ci,
        'std_ci':             std_ci,
        'phoneme':            phon_results,
        'word':               word_results,
        'cluster':            cluster_results,
        'cluster_df':         cluster_df,
        'recall_time':        time_results,
        'confidence':         conf_results,
        'regression':         regression,
        'roc':                roc_df,
        'roc_auc':            roc_auc,
        'correlations':       correlations,
        'counts':             counts,
    }


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
def find_target_folder(input_directory, output_subdir='whisperx_out'):
    """Find all subject directories containing output_subdir results.

    Returns a list of (subject_directory, subject_name) tuples collected
    across every split found under *input_directory*.  Duplicate subjects
    (same name appearing in multiple splits) are kept so that all sessions
    are analysed.
    """
    subject_dirs = []  # list of (subject_directory_path, subject_name)
    seen = set()

    for dirpath, dirnames, filenames in os.walk(input_directory):
        if os.path.basename(dirpath) == output_subdir:
            if all(filename.endswith('.csv') for filename in filenames):
                # dirpath = .../subject/session_N/whisperx_out
                session_directory = os.path.dirname(dirpath)       # .../subject/session_N
                subject_directory = os.path.dirname(session_directory)  # .../subject
                parent_directory  = os.path.dirname(subject_directory)  # dir holding all subjects for this split

                # Collect every subject under this parent (i.e. this split)
                if parent_directory not in seen:
                    seen.add(parent_directory)
                    for name in os.listdir(parent_directory):
                        full = os.path.join(parent_directory, name)
                        if os.path.isdir(full):
                            subject_dirs.append((full, name))

    return subject_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluates outputs of AI annotators based on different metrics')
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to the predictions (.csv) directory.')
    parser.add_argument('--gt_dir', type=str, help='Path to the ground truth data (in ann/par format).')
    parser.add_argument('--verbose', action='store_true', default=False, help='Using this flag will execute all optional print statements for ease with debugging.')
    parser.add_argument('--use_csv', action='store_true', default=False, help='Using this flag will make the program retrieve GT data from a single csv file.')
    parser.add_argument('--csvpath', type=str, default=None, help='Path to the ground truth data (in csv format).')
    parser.add_argument('--output_subdir', type=str, default='whisperx_out',
                        choices=['whisper_out', 'whisperx_out', 'assemblyai_out'],
                        help='Name of the backend output subdirectory to analyze (default: whisperx_out).')
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
    run_all_analysis(args.gt_dir, args.pred_dir, args.verbose, args.use_csv, args.csvpath,
                     output_subdir=args.output_subdir)


