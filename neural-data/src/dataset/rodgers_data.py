import os
import glob
import numpy as np
import pandas as pd
from utils.nwb_utils import get_nwbfile
from tqdm import tqdm

BIN_SIZE = 0.045 # seconds = 45 ms
RODGERS_DATA_DIR = "/data/group_data/neuroagents_lab/neural_datasets/rodgers2022_somatosensory/000231"


def rodgers_make_trials_df(nwbfile, use_six_stimuli=False):
    """
    Create pandas dataframe of trials from nwbfile
    Output dataframe columns: 'id', 'start_time', 'stop_time',
        'first_touch_time', 'choice', 'stimulus', 'choice_time', 'trial'

    By default, the two stimuli are convex and concave (2).
    Set use_six_stimuli=True to use close/medium/far (3) * convex/concave (2) = 6 stimuli
    """
    trials_df = nwbfile.intervals['trials'].to_dataframe()
    # from the paper: Trials for which the field “ignore_trial” is True should be excluded from analysis, typically because they were used for behavioral shaping (such as non-random or direct delivery) or for optogenetic stimulation. Rarely, trials were “munged” (e.g., the motor failed to move the shape, or other experimental glitch), and on such trials the field “ignore_trial” is also set to True.
    trials_df = trials_df[trials_df['ignore_trial'] == False]
    # drop the incorrect trials
    trials_df = trials_df[trials_df['outcome'] == 'correct']
    # make stimulus column
    trials_df = trials_df[trials_df['stimulus'].isin(['convex', 'concave'])]
    if use_six_stimuli:
        trials_df['stimulus'] = trials_df['stimulus'] + ' ' + trials_df['servo_position']
        trials_df = trials_df.drop(columns=['servo_position'])

    contacts_timeseries = _rodgers_make_whisker_contact_timeseries(nwbfile)

    def get_first_touch(row):
        cropped = contacts_timeseries[row['start_time']:row['choice_time']]
        touch_times = cropped.index[cropped]
        return touch_times[0] if len(touch_times) > 0 else np.nan
    trials_df.loc[:, 'first_touch_time'] = trials_df.apply(get_first_touch, axis=1)

    # clean up columns
    trials_df = trials_df.drop(columns=['response_window_open_time', 'direct_delivery', 'optogenetic', 'stim_is_random', 'ignore_trial', 'outcome', 'rewarded_side'])
    # reset row index
    trials_df.reset_index(inplace=True)
    return trials_df


def _rodgers_make_whisker_contact_timeseries(nwbfile):
    WHISKER_NAMES = ["C0", "C1", "C2", "C3"]
    timestamps = []
    for whisker in WHISKER_NAMES:
        whisker_timestamps = nwbfile.processing['behavior']['processed_position_whisker_'+whisker]['angle'].timestamps[:]
        if len(whisker_timestamps) > len(timestamps):
            timestamps = whisker_timestamps

    aligned_whisker_contacts = np.full_like(timestamps, False, dtype=bool)
    for whisker in WHISKER_NAMES:
        try:
            whisker_timestamps = nwbfile.processing['behavior']['processed_position_whisker_'+whisker]['angle'].timestamps[:]
            whisker_contacts = nwbfile.processing['behavior']['contacts_by_whisker_'+whisker]['start_time'].data[:]
            idx = np.searchsorted(timestamps, whisker_contacts)
            aligned_whisker_contacts[idx] = True
        except KeyError:
            continue

    return pd.Series(data=aligned_whisker_contacts, index=timestamps)


def rodgers_get_spikes(nwbfile):
    """
    Extract spikes from nwbfile.
    """
    spike_df = nwbfile.units[:]
    # adjust spike times to be relative to trial start time
    obs_starttime = nwbfile.acquisition['extracellular array recording'].starting_time
    spikes = spike_df.spike_times - obs_starttime
    return spikes


def rodgers_bin_spikes(trials_df, spikes, bin_size=BIN_SIZE):
    """
    Bins spikes into 45 ms bins, from trial start time to the next bin size including trial stop time.
    Output shape: (n_trials, n_stimuli, n_units)
    """
    n_units = len(spikes)
    n_trials = len(trials_df)
    stimulus_map = {stimulus: idx for idx, stimulus in enumerate(trials_df['stimulus'].unique())}

    data = np.zeros((n_trials, len(stimulus_map), n_units))

    for trial_i, trial in trials_df.iterrows():
        # start = np.where(np.isnan(trial['first_touch_time']), trial['start_time'], trial['first_touch_time'])
        start = trial['start_time']
        stop = trial['choice_time']
        stimulus = trial['stimulus']

        stimulus_i = stimulus_map[stimulus]
        time_bins = np.arange(start, stop + bin_size, bin_size)

        for unit_i, (_, spike_times) in enumerate(spikes.items()):
            spikes_in_trial = spike_times[(spike_times >= start) & (spike_times < stop)]
            binned_spikes, _ = np.histogram(spikes_in_trial, bins=time_bins)
            data[trial_i, stimulus_i, unit_i] = binned_spikes.mean() / bin_size
    return data


def load_rodgers_data(path=RODGERS_DATA_DIR, use_six_stimuli=False):
    """
    Input path: npz file OR directory containing nwb files
    Output data is dict { animal: [sessions] } where
        a session is a numpy array of shape (n_trials, n_stimuli, n_units)
    """
    if path.endswith(".npz"):
        data_dict = dict(np.load(path, allow_pickle=True))
    else:
        filenames = glob.glob(path + "/**/*behavior+ecephys+image.nwb")
        data_dict = {}
        for filename in tqdm(filenames, desc="Loading files"):
            name = os.path.basename(filename).split("_behavior")[0]
            animal = name.split("_")[0]
            nwbfile = get_nwbfile(filename)
            trials_df = rodgers_make_trials_df(nwbfile, use_six_stimuli=use_six_stimuli)
            spikes = rodgers_get_spikes(nwbfile)
            spike_data = rodgers_bin_spikes(trials_df, spikes)
            if animal in data_dict:
                data_dict[animal].append(spike_data)
            else:
                data_dict[animal] = [spike_data]
        data_dict = _concatenate_rodgers_data(data_dict)
        
    return data_dict


def concatenate_sessions(sessions):
    """
    Concatenate sessions for each animal by setting last dimension
    (neurons per session) to (total neurons over all sessions) and padding with NaN.
    Output: np.array((max_num_trials, n_stimuli, total_units))
    Example:
        sessions[0].shape == (100, 2, 30)
        sessions[1].shape == (150, 2, 35)
        output[0:100, :, 0:30] = sessions[0]
        output[0:150, :, 30:65] = sessions[1]
    """
    num_stimuli = sessions[0].shape[1]
    assert all(session.shape[1] == num_stimuli for session in sessions), "All sessions should have the same number of stimuli"

    unit_counts = [session.shape[2] for session in sessions]
    total_units = np.sum(unit_counts)
    max_trials = max([session.shape[0] for session in sessions])
    concatenated_sessions = np.full((max_trials, num_stimuli, total_units), np.nan)

    start_units = 0
    for session in sessions:
        num_trials, _, num_units = session.shape
        concatenated_sessions[:num_trials, :, start_units:start_units+num_units] = session
        start_units += num_units

    return concatenated_sessions


def concat_other_animals(animal_id, data_per_animal):
    """
    Concatenate data from all animals except the one specified by animal_id.
    """
    rest_of_animals = data_per_animal.copy()
    del rest_of_animals[animal_id]
    return concatenate_sessions(list(rest_of_animals.values()))


def _concatenate_rodgers_data(rodgers_data):
    """
    Concatenate sessions for each animal by setting last dimension
    (neurons per session) to (total neurons over all sessions) and padding with NaN.
    Output: dict {animal: np.array((total_trials, n_stimuli, total_units))}
    """
    data_per_animal = {}
    for animal, sessions in rodgers_data.items():
        concatenated_sessions = concatenate_sessions(sessions)
        data_per_animal[animal] = concatenated_sessions
    return data_per_animal


def load_linregress_results(filepath):
    """
    Load linear regression results from a .npz file (generated by animal_to_animal_consistency).
    Output: dict {animal_id: {split: {r_: data}}}
            The shape of data is (num_bootstrap_iters aka num_splithalves, num_train_test_splits, num_units)
    """
    results = np.load(filepath, allow_pickle=True)
    return {key: value.item() for key, value in results.items()}


if __name__ == "__main__":
    rodgers6_data = load_rodgers_data(use_six_stimuli=True)
    np.savez("./data/rodgers6_data.npz", **rodgers6_data, allow_pickle=True)
