import matplotlib

matplotlib.use('TkAgg')
import os
import matplotlib.pyplot as plt
from astropy.visualization import hist
import numpy as np
from pynwb import NWBHDF5IO
import pandas as pd
from bisect import bisect_left, bisect_right
import re
from PIL import Image


# file_name = 'sub-anm106211/sub-anm106211_ses-20100925_behavior+icephys.nwb'
class LowNoiseEncodingTrial:
    def __init__(self, file_name):

        self.nwbfile = NWBHDF5IO(file_name, 'r').read()

        self.file_name = file_name

        # initialization
        self.start_time, self.end_time, self.go_hit_trials, self.nogo_cr_trials = self.filter_trials()

    def filter_trials(self):
        # get trials
        trials = self.nwbfile.intervals['trials']

        # print(trials.colnames)  # Lists the column names in the trials table
        # ('start_time', 'stop_time', 'type', 'response', 'stim_present', 'pole_position', 'first_lick_time',
        # 'pole_in_time', 'pole_out_time')

        start_time = trials['start_time'].data[:][0]
        end_time = trials['stop_time'].data[:][-1]

        trial_data = {col: trials[col].data[:] for col in trials.colnames}
        trials_df = pd.DataFrame(trial_data)

        # Filter for (Go, Hit)
        go_hit_trials = trials_df[(trials_df['type'] == 'Go') & (trials_df['response'] == 'Hit')]  # TP
        # Filter for (Nogo, CR)
        nogo_cr_trials = trials_df[(trials_df['type'] == 'Nogo') & (trials_df['response'] == 'CR')]  # TN
        return start_time, end_time, go_hit_trials, nogo_cr_trials

    # @staticmethod
    # def find_data_interval(time_ticks, time_pairs, data):
    #     """
    #     For each (start, end) pair, find the indices in time_ticks where the times fit.
    #     :param time_ticks: List of time tick values (sorted in ascending order).
    #     :param time_pairs: List of (start, end) pairs.
    #     :return: List of (start_index, end_index) pairs, where start_index and end_index correspond
    #              to the position in time_ticks for each (start, end).
    #     """
    #     indices = []
    #     intervals = []
    #     for start, end in time_pairs:
    #         # Find the index in time_ticks where 'start' would fit
    #         start_index = bisect_right(time_ticks, start) - 1
    #         # Find the index in time_ticks where 'end' would fit
    #         end_index = bisect_left(time_ticks, end)
    #
    #         # Adjust the indices for edge cases
    #         if start < time_ticks[0]:
    #             start_index = 0  # reset to the first tick
    #         if end >= time_ticks[-1]:
    #             end_index = len(time_ticks) - 1  # reset to the last tick
    #
    #         interval = data[start_index:end_index + 1]
    #         if np.isnan(interval).any():
    #             print(f'Caution! current interval contains NaN, skipping...')
    #         else:
    #             indices.append((start_index, end_index))
    #             intervals.append(interval)
    #
    #     return indices, intervals

    @staticmethod
    def find_data_interval(time_ticks, time_pairs, data):
        """
        For each (start, end) pair, find the indices in time_ticks where the interval is contained within (start, end).
        :param time_ticks: List of time tick values (sorted in ascending order).
        :param time_pairs: List of (start, end) pairs.
        :param data: List or array of data values corresponding to time_ticks.
        :return: List of (start_index, end_index) pairs, where start_index and end_index correspond
                 to the position in time_ticks for each (start, end).
        """
        indices = []
        intervals = []

        nan_type_dict = {'no_nan': 0, 'start_or_end': 0, 'middle': 0, 'scattered': 0}

        for start, end in time_pairs:
            # Find the index in time_ticks where 'start' would fit or the next greater value
            start_index = bisect_left(time_ticks, start)
            # Find the index in time_ticks where 'end' would fit or the previous smaller value
            end_index = bisect_right(time_ticks, end) - 1

            # Adjust the indices for edge cases
            if start < time_ticks[0]:
                start_index = 0  # reset to the first tick
            if end >= time_ticks[-1]:
                end_index = len(time_ticks) - 1  # reset to the last tick
            interval = data[start_index:end_index + 1]

            if np.isnan(interval).any():
                # Check positions of NaNs
                nan_mask = np.isnan(interval)
                first_nan = np.argmax(nan_mask)
                last_nan = len(interval) - np.argmax(nan_mask[::-1]) - 1

                if np.all(nan_mask[first_nan:last_nan + 1]):
                    # if True, meaning there is no real number within the first & last NaN
                    if first_nan == 0 or last_nan == len(interval) - 1:
                        nan_type = 'start_or_end'
                        if first_nan == 0:  # nan in the beginning, adjust the start_index
                            start_index = start_index + last_nan + 1
                        else:  # last_nan == len(interval) - 1
                            end_index = end_index - ((last_nan - first_nan) + 1)
                        assert end_index - start_index >= 100, f'currently chopped interval starts from {start_index} to {end_index}'
                        indices.append((start_index, end_index))
                        interval = interval[~nan_mask]
                        intervals.append(interval)
                    else:
                        nan_type = 'middle'
                else:  # if False, meaning there is a real number within the first & last NaN
                    nan_type = 'scattered'
            else:
                indices.append((start_index, end_index))
                intervals.append(interval)

                nan_type = 'no_nan'
            nan_type_dict[nan_type] += 1
        return indices, intervals, nan_type_dict

    @staticmethod
    def are_equal_with_precision(A, B):
        """Check if B matches A when rounded to the same decimal precision as A."""

        # Get the decimal places of A
        def decimal_places(number):
            """Calculate the number of decimal places of a number."""
            # Convert number to string and split on the decimal point
            str_num = str(number)
            if '.' in str_num:
                return len(str_num.split('.')[1])
            return 0

        decimal_digits = decimal_places(A)
        # Round B to match A's decimal precision
        B_rounded = round(B, decimal_digits)
        # Check for equality
        return A == B_rounded

    def get_behavior_data(self, field, is_behavior=True):
        if is_behavior:
            behavior_data = self.nwbfile.acquisition['behavior']
            # print("Available data series in behavior:", list(behavior_data.time_series.keys()))
            # ['amplitude', 'beam_break_times', 'delta_kappa', 'distance_to_pole', 'phase', 'pole_available',
            # 'set_point', 'theta_at_base', 'theta_filt', 'touch_offset', 'touch_onset']

            data = behavior_data.time_series[field].data[:]
            timestamps = behavior_data.time_series[field].timestamps[:]
        else:
            data = self.nwbfile.acquisition[field].data[:]
            timestamps = self.nwbfile.acquisition[field].timestamps[:]
        return data, timestamps

    def make_time_series(self, field, is_behavior=True):  # 'theta_at_base'
        data, timestamps = self.get_behavior_data(field=field, is_behavior=is_behavior)

        if not self.are_equal_with_precision(A=self.start_time, B=timestamps[0]):
            print(f'start time in trials={self.start_time} & behavior data={timestamps[0]} is different')
            assert 0 <= (timestamps[0] - self.start_time) <= 0.01
        if not self.are_equal_with_precision(A=self.end_time, B=timestamps[-1]):
            print(f'end time in trials={self.end_time} & behavior data={timestamps[-1]} is different')
            assert 0 <= (self.end_time - timestamps[-1]) <= 0.01

        nogo_cr_pole_times = self.nogo_cr_trials[['pole_in_time', 'pole_out_time']].to_numpy()
        go_hit_pole_times = self.go_hit_trials[['pole_in_time', 'pole_out_time']].to_numpy()

        nogo_cr_indices, nogo_cr_intervals, nogo_nan_dict = self.find_data_interval(time_ticks=timestamps,
                                                                                    time_pairs=nogo_cr_pole_times,
                                                                                    data=data)
        go_hit_indices, go_hit_intervals, go_nan_dict = self.find_data_interval(time_ticks=timestamps,
                                                                                time_pairs=go_hit_pole_times,
                                                                                data=data)

        return go_hit_intervals, nogo_cr_intervals, go_hit_indices, nogo_cr_indices, go_nan_dict, nogo_nan_dict


def plot_time_series(go_hit_intervals, nogo_cr_intervals, go_hit_indices, nogo_cr_indices, file_path, title):
    all_indices = nogo_cr_indices + go_hit_indices
    min_x = min(start for start, _ in all_indices)
    max_x = max(end for _, end in all_indices)
    x_range = max_x - min_x

    # Plotting the data
    if nogo_cr_intervals is not None and go_hit_intervals is not None:
        plt.figure(figsize=(x_range // 10000, 6))

    # Plot nogo_cr_intervals in black
    if nogo_cr_intervals is not None:
        for i, interval in enumerate(nogo_cr_intervals):
            start, end = nogo_cr_indices[i]
            plt.plot(range(start, end + 1), interval, color='black', label='nogo_cr' if i == 0 else "")

    # Plot go_hit_intervals in blue
    if go_hit_intervals is not None:
        for i, interval in enumerate(go_hit_intervals):
            start, end = go_hit_indices[i]
            plt.plot(range(start, end + 1), interval, color='blue', label='go_hit' if i == 0 else "")

    # Adding labels, title, and legend
    # plt.xlim(min_x, max_x)
    plt.xlabel('Time Tick')
    plt.ylabel('Value')
    plt.title(f"{title}-nogo={len(nogo_cr_intervals) if nogo_cr_intervals is not None else None}"
              f"-go={len(go_hit_intervals) if go_hit_intervals is not None else None}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file_path}/{title}.png", bbox_inches='tight', )
    plt.close()
    img = Image.open(f'{file_path}/{title}.png')
    return img


def find_test_date(dir):  # sub-anm106211_ses-20100925_behavior+icephys.nwb
    # Regular expression to find the date in the format YYYYMMDD
    date_match = re.search(r'ses-(\d{8})', dir)

    # Extract and print the date if found
    if date_match:
        date = date_match.group(1)
    else:
        date = None
    return date


def plot_histogram(arr, file_path, title, bin_size=None):
    # Group data in chunks of bin_size and compute average for each bin
    # averaged_data = arr if bin_size is None else [np.mean(arr[i:i + bin_size]) for i in range(0, len(arr), bin_size)]
    if bin_size is not None:
        num_bins = len(arr) // bin_size + (1 if len(arr) % bin_size != 0 else 0)
    else:
        num_bins = 'freedman'

    # Plot the histogram
    hist(arr, bins=num_bins)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.savefig(f'{file_path}/{title}.png', bbox_inches='tight', )
    plt.close()
    img = Image.open(f'{file_path}/{title}.png')
    return img


def visualize_nxm_plots(img_list, n, m, file_path, title):
    """
    Visualize a list of images in an n x m grid with minimal spacing and save as a single image.

    Parameters:
    img_list: List of PIL Image objects to display
    n: Number of rows in the grid
    m: Number of columns in the grid
    file_path: Path to save the combined grid image
    title: Title of the output file
    """
    if len(img_list) != n * m:
        raise ValueError("The number of images must match n * m.")

    # Ensure all images are the same size
    # Optionally, you can resize images to a standard size
    widths, heights = zip(*(im.size for im in img_list))
    max_width = max(widths)
    max_height = max(heights)
    standard_size = (max_width, max_height)
    img_list = [im.resize(standard_size) for im in img_list]

    # Create a new blank image with the appropriate size
    total_width = m * max_width
    total_height = n * max_height
    new_im = Image.new('RGB', (total_width, total_height))

    # Paste images into the grid without any spacing
    for idx, im in enumerate(img_list):
        row = idx // m
        col = idx % m
        x_offset = col * max_width
        y_offset = row * max_height
        new_im.paste(im, (x_offset, y_offset))

    # Save the combined image
    combined_path = f"{file_path}/{title}.png"
    new_im.save(combined_path)

    # Optional: Display the final combined image
    # new_im.show()


subject = 'sub-anm106211'
file_name = 'sub-anm106211_ses-20100925_behavior+icephys.nwb'
inst = LowNoiseEncodingTrial(file_name=f'{subject}/{file_name}')
test_date = find_test_date(dir=file_name)

vis_dir = f'neural_data_vis/{subject}/{test_date}'
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

# go_hit_intervals, nogo_cr_intervals, go_hit_indices, nogo_cr_indices, go_nan_dict, nogo_nan_dict = inst.make_time_series(
#     field='theta_at_base')
#
# plot_time_series(go_hit_intervals=go_hit_intervals, nogo_cr_intervals=nogo_cr_intervals,
#                  go_hit_indices=go_hit_indices, nogo_cr_indices=nogo_cr_indices,
#                  file_path=vis_dir, title='Nogo-cr & Go-hit time series (theta) visualization')
#
# bin_size = None
# go_list = [plot_histogram(go_hit_intervals[inst_id], file_path=vis_dir,
#                           title=f'go_hit.inst={inst_id}.bin_size={bin_size}', bin_size=bin_size) for inst_id in
#            range(3)]
# nogo_list = [plot_histogram(nogo_cr_intervals[inst_id], file_path=vis_dir,
#                             title=f'nogo_cr.inst={inst_id}.bin_size={bin_size}', bin_size=bin_size) for inst_id in
#              range(3)]
#
# visualize_nxm_plots(img_list=go_list + nogo_list, n=2, m=3, file_path=vis_dir,
#                     title=f'combined_vis.bin_size={bin_size}')


go_hit_intervals, nogo_cr_intervals, go_hit_indices, nogo_cr_indices, go_nan_dict, nogo_nan_dict = inst.make_time_series(
    field='theta_at_base')

go_list = [plot_time_series(go_hit_intervals=[go_hit_intervals[inst_id]],
                            go_hit_indices=[go_hit_indices[inst_id]],
                            nogo_cr_intervals=None, nogo_cr_indices=[],
                            title=f'go_hit.neural_data.inst={inst_id}', file_path=vis_dir) for inst_id in range(3)]
nogo_list = [plot_time_series(go_hit_intervals=None, go_hit_indices=[],
                              nogo_cr_intervals=[nogo_cr_intervals[inst_id]],
                              nogo_cr_indices=[nogo_cr_indices[inst_id]],
                              title=f'nogo_cr.neural_data.inst={inst_id}', file_path=vis_dir) for inst_id in range(3)]

visualize_nxm_plots(img_list=go_list + nogo_list, n=2, m=3, file_path=vis_dir,
                    title=f'combined_vis.neural_data')
exit()

path = "/data/group_data/neuroagents_lab/neural_datasets/somatosensory_cortex"


# ['sub-anm234231', 'sub-anm217330', 'sub-anm211295', 'sub-anm106213', 'sub-anm234232', 'sub-anm226695',
# 'sub-anm244025', 'sub-anm134333', 'sub-anm244028', 'sub-anm226692', 'sub-anm17702', 'sub-jf42400',
# 'sub-anm131970', 'sub-anm234230', 'sub-anm266951', 'sub-jf37166', 'sub-anm106211', 'sub-anm101105',
# 'sub-anm226694', 'sub-anm266945', 'sub-anm215592', 'sub-anm244024', 'sub-jf35892']


def list_files(dir_path):
    file_names = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_names.append(file)
    return file_names


go_dict = {'max': [], 'min': [], 'delta': [], 'no_nan': 0, 'start_or_end': 0, 'middle': 0, 'scattered': 0}
nogo_dict = {'max': [], 'min': [], 'delta': [], 'no_nan': 0, 'start_or_end': 0, 'middle': 0, 'scattered': 0}

for subject in os.listdir(path):
    if not os.path.isdir(os.path.join(path, subject)):
        print(f'{os.path.join(path, subject)} is not a valid dir for neural data')
        continue
    file_names = list_files(dir_path=os.path.join(path, subject))
    for file_name in file_names:
        inst = LowNoiseEncodingTrial(file_name=f'{path}/{subject}/{file_name}')
        go_hit_intervals, nogo_cr_intervals, _, _, go_nan_dict, nogo_nan_dict = inst.make_time_series(
            field='theta_at_base')

        for key in ['no_nan', 'start_or_end', 'middle', 'scattered']:
            go_dict[key] += go_nan_dict[key]
            nogo_dict[key] += nogo_nan_dict[key]

        go_max = [max(interval) for interval in go_hit_intervals]
        go_min = [min(interval) for interval in go_hit_intervals]
        go_delta = [max(interval) - min(interval) for interval in go_hit_intervals]

        nogo_max = [max(interval) for interval in nogo_cr_intervals]
        nogo_min = [min(interval) for interval in nogo_cr_intervals]
        nogo_delta = [max(interval) - min(interval) for interval in nogo_cr_intervals]

        go_dict['max'] = go_dict['max'] + go_max
        go_dict['min'] = go_dict['min'] + go_min
        go_dict['delta'] = go_dict['delta'] + go_delta

        nogo_dict['max'] = nogo_dict['max'] + nogo_max
        nogo_dict['min'] = nogo_dict['min'] + nogo_min
        nogo_dict['delta'] = nogo_dict['delta'] + nogo_delta

for key in ['no_nan', 'start_or_end', 'middle', 'scattered']:
    print(key, 'go', go_dict[key], 'nogo', nogo_dict[key])

vis_dir = f'neural_data_vis'
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)
bin_size = None
go_list = [plot_histogram(np.array(val), file_path=vis_dir,
                          title=f'go_hit.{key}.bin_size={bin_size}', bin_size=bin_size) for key, val in
           go_dict.items() if key not in ['no_nan', 'start_or_end', 'middle', 'scattered']]
nogo_list = [plot_histogram(np.array(val), file_path=vis_dir,
                            title=f'nogo_cr.{key}.bin_size={bin_size}', bin_size=bin_size) for key, val in
             nogo_dict.items() if key not in ['no_nan', 'start_or_end', 'middle', 'scattered']]

visualize_nxm_plots(img_list=go_list + nogo_list, n=2, m=3, file_path=vis_dir,
                    title=f'overall.max.min.delta.bin_size={bin_size}')

# make pole_in < S < E < pole_out [finished]
# visualize the go-hit & nogo-cr time series of one test-trial (visualize NaN as well)
# then decide what to do with the NaN (compute the portion of the NaN)
# solution: discard & remove & interpolate
