import json
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Source: http://ethen8181.github.io/machine-learning/model_selection/auc/auc.html#Implementation
def _binary_clf_curve(y_true, y_score):
    """
    Calculate true and false positives per binary classification
    threshold (can be used for roc curve or precision/recall curve);
    the calcuation makes the assumption that the positive case
    will always be labeled as 1

    Parameters
    ----------
    y_true : 1d ndarray, shape = [n_samples]
        True targets/labels of binary classification

    y_score : 1d ndarray, shape = [n_samples]
        Estimated probabilities or scores

    Returns
    -------
    tps : 1d ndarray
        True positives counts, index i records the number
        of positive samples that got assigned a
        score >= thresholds[i].
        The total number of positive samples is equal to
        tps[-1] (thus false negatives are given by tps[-1] - tps)

    fps : 1d ndarray
        False positives counts, index i records the number
        of negative samples that got assigned a
        score >= thresholds[i].
        The total number of negative samples is equal to
        fps[-1] (thus true negatives are given by fps[-1] - fps)

    thresholds : 1d ndarray
        Predicted score sorted in decreasing order

    References
    ----------
    Github: scikit-learn _binary_clf_curve
    - https://github.com/scikit-learn/scikit-learn/blob/ab93d65/sklearn/metrics/ranking.py#L263
    """

    # sort predicted scores in descending order
    # and also reorder corresponding truth values
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically consists of tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve
    distinct_indices = np.where(np.diff(y_score))[0]
    end = np.array([y_true.size - 1])
    threshold_indices = np.hstack((distinct_indices, end))

    thresholds = y_score[threshold_indices]
    tps = np.cumsum(y_true)[threshold_indices]

    # (1 + threshold_indices) = the number of positives
    # at each index, thus number of data points minus true
    # positives = false positives
    fps = (1 + threshold_indices) - tps
    return tps, fps, thresholds


def _roc_auc_score(y_true, y_score):
    """
    Compute Area Under the Curve (AUC) from prediction scores

    Parameters
    ----------
    y_true : 1d ndarray, shape = [n_samples]
        True targets/labels of binary classification

    y_score : 1d ndarray, shape = [n_samples]
        Estimated probabilities or scores

    Returns
    -------
    auc : float
    """

    # ensure the target is binary
    if np.unique(y_true).size != 2:
        raise ValueError('Only two class should be present in y_true. ROC AUC score '
                         'is not defined in that case.')

    tps, fps, _ = _binary_clf_curve(y_true, y_score)

    # convert count to rate
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    # compute AUC using the trapezoidal rule;
    # appending an extra 0 is just to ensure the length matches
    zero = np.array([0])
    tpr_diff = np.hstack((np.diff(tpr), zero))
    fpr_diff = np.hstack((np.diff(fpr), zero))
    auc = np.dot(tpr, fpr_diff) + np.dot(tpr_diff, fpr_diff) / 2
    return auc

def _calculate_pref_angle_and_auroc_pvalues(neuron_data, timestep_from, timestep_to, permutation_n=1000):
    """
        For a given neuron (structure of neuron_data has to be
        (n_contidions, number_of_trials, number_of_timesteps)),
        average its firing rate across the window [timestep_from, timestep_to],
        and calculate its preferred stimulus angle and the associated p-values of AUROC.
        p-values are calculated separately for each pair of opposite directions
        permutation_n is how many permutations to do for the permutation test
    """
    n_conditions = len(neuron_data)
    condition_thetas = np.arange(n_conditions)/n_conditions * 2 * np.pi
    mean_window_frs = []  # mean_window_frs will be (n_contidions, number_of_trials)
    for j in range(n_conditions):
        mean_window_frs.append(np.mean(neuron_data[j][:,timestep_from:timestep_to], axis=1))    
    mean_frs = np.array([np.mean(condition) for condition in mean_window_frs])  # mean firing rate for each condition
    
    complex_vector = np.sum(np.exp(1j*condition_thetas)*mean_frs)/np.sum(mean_frs)
    dsi = np.abs(complex_vector)  # direction selectivity index (DSI)
    pref_angle = np.angle(complex_vector)

    pvalues = []
    for i in range(n_conditions//2):  # for every pair of opposite directions (i, j)
        j = i + n_conditions//2

        trials_i = mean_window_frs[i]
        trials_j = mean_window_frs[j]

        y_true = np.array([1]*len(trials_i) + [0]*len(trials_j))
        y_scores = np.concatenate((trials_i, trials_j))
        auc_score = _roc_auc_score(y_true, y_scores)

        null_auc_scores_distribution = []  # distribution of AUROC scores for reshuffled trial labels
        for p in range(permutation_n):
            np.random.shuffle(y_true)
            null_auc_scores_distribution.append(_roc_auc_score(y_true, y_scores))

        pvalue = np.sum(auc_score<np.array(null_auc_scores_distribution))/permutation_n
        pvalue = min(pvalue, 1-pvalue)
        pvalues.append(pvalue)
    return pref_angle, pvalues


def _reshuffle_trial_labels(neuron_data):
    """
        Given an array of size (n_conditions, number_of_trials, ...)
        reshuffle the trial labels and return an array of the same structure
    """
    n_conditions = len(neuron_data)
    combined_neuron_data = np.concatenate(neuron_data)
    np.random.shuffle(combined_neuron_data)
    neuron_data_reshuffled = []  # group the trials back with the same n per condition
    i_from = 0
    for i in range(n_conditions):
        i_to = i_from+len(neuron_data[i])
        neuron_data_reshuffled.append(combined_neuron_data[i_from:i_to])
        i_from = i_to
    return neuron_data_reshuffled

    
def _calculate_pref_angle_and_dsi_pvalue(neuron_data, timestep_from, timestep_to, permutation_n=1000):
    """
        For a given neuron (structure of neuron_data has to be
        (n_contidions, number_of_trials, number_of_timesteps)),
        average its firing rate across the window [timestep_from, timestep_to],
        and calculate its preferred stimulus angle and the associated p-values of DSI.
        permutation_n is how many permutations to do for the permutation test
    """
    n_conditions = len(neuron_data)
    condition_thetas = np.arange(n_conditions)/n_conditions * 2 * np.pi
    mean_window_frs = []  # mean_window_frs will be (n_contidions, number_of_trials)
    for j in range(n_conditions):
        mean_window_frs.append(np.mean(neuron_data[j][:,timestep_from:timestep_to], axis=1))    
    mean_frs = np.array([np.mean(condition) for condition in mean_window_frs])  # mean firing rate for each condition
    
    complex_vector = np.sum(np.exp(1j*condition_thetas)*mean_frs)/np.sum(mean_frs)
    dsi = np.abs(complex_vector)  # direction selectivity index (DSI)
    pref_angle = np.angle(complex_vector)

    null_dsi_distribution = []  # distribution of DSIs for reshuffled trial labels
    for p in range(permutation_n):
        mean_window_frs_reshuffled = _reshuffle_trial_labels(mean_window_frs)
        mean_frs_reshuffled = np.array([np.mean(condition) for condition in mean_window_frs_reshuffled])
        complex_vector = np.sum(np.exp(1j*condition_thetas)*mean_frs_reshuffled)/np.sum(mean_frs_reshuffled)
        null_dsi_distribution.append(np.abs(complex_vector))  # direction selectivity index (DSI)
    pvalue = np.sum(dsi<np.array(null_dsi_distribution))/permutation_n
    return pref_angle, pvalue

def calculate_pref_angle_and_dsis(neuron_data, timestep_from, timestep_to):
    """
        For a given neuron (structure of neuron_data has to be
        (n_contidions, number_of_trials, number_of_timesteps)),
    """
    n_conditions = len(neuron_data)
    condition_thetas = np.arange(n_conditions)/n_conditions * 2 * np.pi
    mean_window_frs = []  # mean_window_frs will be (n_contidions, number_of_trials)
    for j in range(n_conditions):
        mean_window_frs.append(np.mean(neuron_data[j][:,timestep_from:timestep_to], axis=1))    
    mean_frs = np.array([np.mean(condition) for condition in mean_window_frs])  # mean firing rate for each condition
    
    complex_vector = np.sum(np.exp(1j*condition_thetas)*mean_frs)/np.sum(mean_frs)
    dsi = np.abs(complex_vector)  # direction selectivity index (DSI)
    pref_angle = np.angle(complex_vector)

    return pref_angle, dsi


def determine_neuron_selectivities(get_neuron_data_for_neuron, N_neurons, n_conditions, cue_from, cue_to, delay_from, delay_to,
                                   use_auroc=False, reshuffle_trials=False, permutation_n=1000, verbose=True):
    """
        For every neuron in the dataset, compute its preferred condition
        and the associated p-value, for the cue and delay periods.

        get_neuron_data_for_neuron: callable - the function that gets neuron data from the 
            relevant file (see implementations). This function will be specific to the dataset.
            neuron_data is returned of shape (n_conditions, number_of_trials, number_of_timesteps)
        N_neurons is number of neurons in the dataset
        n_conditions is #stimulus directions (e.g. up, down, left, right = 4)
        cue_from, cue_to, delay_from, delay_to - bin intervals that determine cue/delay
            
        if use_auroc, uses AUROC for statistical tests; else, uses the absolute value of DSI
        if reshuffle_trials, reshuffles all trial labels before doing any analysis 
        (this is used to calculate type I error later)

        returns (for cue and delay periods):
            pval_cue, pval_delay - p-values - of shape (N_neurons, )
            pref_angle_cue, pref_angle_delay - the preferred angls - of shape (N_neurons, )
            n_trials - number of trials per condition, shape (N_neurons, n_conditions)
    """
    pval_cue = np.zeros((N_neurons, ), dtype=float)
    pval_delay = np.zeros((N_neurons, ), dtype=float)
    pref_angle_cue = np.zeros((N_neurons, ), dtype=float)
    pref_angle_delay = np.zeros((N_neurons, ), dtype=float)
    n_trials = np.zeros((N_neurons, n_conditions))  # how many trials per condition
    report_every = N_neurons//20
    for i in range(N_neurons):
        if verbose and i%report_every==0: print(f"{i}/{N_neurons}", end=" ")
        neuron_data = get_neuron_data_for_neuron(i)
        n_trials[i] = [len(nd) for nd in neuron_data] # how many trials per condition for this neuron

        if reshuffle_trials:
            neuron_data = _reshuffle_trial_labels(neuron_data)
        
        # cue
        if use_auroc:
            pref_angle, pvalues = _calculate_pref_angle_and_auroc_pvalues(neuron_data, cue_from, cue_to, permutation_n=permutation_n)
            pvalue = min(pvalues)
        else:
            pref_angle, pvalue = _calculate_pref_angle_and_dsi_pvalue(neuron_data, cue_from, cue_to, permutation_n=permutation_n)
        pref_angle_cue[i] = pref_angle
        pval_cue[i] = pvalue
        
        # delay
        if use_auroc:
            pref_angle, pvalues = _calculate_pref_angle_and_auroc_pvalues(neuron_data, delay_from, delay_to, permutation_n=permutation_n)
            pvalue = min(pvalues)
        else:
            pref_angle, pvalue = _calculate_pref_angle_and_dsi_pvalue(neuron_data, delay_from, delay_to, permutation_n=permutation_n)
        pref_angle_delay[i] = pref_angle
        pval_delay[i] = pvalue
    return pval_cue, pval_delay, pref_angle_cue, pref_angle_delay, n_trials


def determine_selective_neuron_groups(pval_cue, pval_delay, n_trials, min_n_trials=5, alpha_threshold=0.05):
    """
        Inputs:
            pval_cue, pval_delay are of shape (N_neurons, ) - see outputs of "determine_neuron_selectivities"
            n_trials of shape (N_neurons, n_conditions) - see outputs of "determine_neuron_selectivities"
        any neuron which has <min_n_trials recorded per condition gets discarded
        Returns cue_selective, delay_selective, both_selective - lists of indices where p values < alpha_threshold
    """
    N_neurons = len(pval_cue)
    cue_selective = []
    delay_selective = []
    for i in range(N_neurons):
        if np.any(n_trials[i]<min_n_trials): continue
        if pval_cue[i] < alpha_threshold: cue_selective.append(i)
        if pval_delay[i] < alpha_threshold: delay_selective.append(i)
    both_selective = [x for x in cue_selective if x in delay_selective]
    return cue_selective, delay_selective, both_selective 


def get_mean_windowed_firing_rates_for_cue_and_delay(get_neuron_data_for_neuron, N_neurons, n_conditions, cue_from, cue_to, delay_from, delay_to, neuron_indices):
    """
        Inputs:
            get_neuron_data_for_neuron: callable - the function that gets neuron data from the 
                relevant file (see implementations). This function will be specific to the dataset.
                neuron_data is returned of shape (n_contidions, number_of_trials, number_of_timesteps)
            N_neurons is number of neurons in the dataset
            n_conditions is #stimulus directions (e.g. up, down, left, right = 4)
            cue_from, cue_to, delay_from, delay_to - bin intervals that determine cue/delay
            neuron_indices - list of indices to get firing rates of
    
        For a given list of indices (neuron_indices) of length M,
        returns a structure mean_firing_rates of shape (n_conditions, 2, M)
    """
    mean_firing_rates = np.zeros((n_conditions, 2, len(neuron_indices)))
    for i, neuron_index in enumerate(neuron_indices):
        neuron_data = get_neuron_data_for_neuron(neuron_index)  # (n_conditions, number_of_trials, number_of_timesteps)
        for j in range(n_conditions):
            mean_firing_rates[j, 0, i] = np.mean(neuron_data[j][:, cue_from:cue_to])
            mean_firing_rates[j, 1, i] = np.mean(neuron_data[j][:, delay_from:delay_to])
    return mean_firing_rates


def show_neuron(get_neuron_data_for_neuron, ax, neuron_index, time_bins, colors=None, convolve=True, convolve_sigma=1.5):
    """
        On a matplotlib axis ax, plot neuron firing rate of 
        the neuron with index neuron_index
        Does not do any formatting of the graph. Just plots the lines
        get_neuron_data_for_neuron: callable - the function that gets neuron data from the 
            relevant file (see implementations). This function will be specific to the dataset.
            neuron_data is returned of shape (n_conditions, number_of_trials, number_of_timesteps)
        time_bins is the bins (in ms) corresponding to timesteps in neuron_data
        Use "convolve" parameter and "convolve_sigma" for smoothing
    """
    neuron_data = get_neuron_data_for_neuron(neuron_index)
    n_conditions = len(neuron_data)
    if colors is None:
        colors = plt.cm.cool(np.linspace(0, 1, n_conditions))
    n_trials = [len(d) for d in neuron_data]
     
    # convolve
    if convolve:
        sigma = convolve_sigma
        dx = 1
        gx = np.arange(-3*sigma, 3*sigma, dx)
        gaussian = np.exp(-(gx/sigma)**2/2)/sigma/(2*np.pi)**0.5
        for j in range(n_conditions):
            for t in range(len(neuron_data[j])):
                neuron_data[j][t] = np.convolve(neuron_data[j][t], gaussian, mode="same")

    for j in range(n_conditions):
        color = colors[j]
        mean = np.mean(neuron_data[j], axis=0)
        sem = np.std(neuron_data[j], axis=0) / (n_trials[j]-1)**0.5
        ax.fill_between(time_bins, mean-sem, mean+sem, color=color, alpha=0.3, linewidth=0)
        ax.plot(time_bins, mean, "-", linewidth=3, color=color, label=f"{j}")