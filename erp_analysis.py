"""
analyze EEG data

Created by Dirk van Moorselaar on 01-07-2015.
Copyright (c) 2015 DvM. All rights reserved.
"""

from scipy.signal import butter, lfilter, freqz
import pylab as pl
import os


f_name = os.path.join('/Users','Dirk','Dropbox','Experiment_data','data','load_accessory','processed_eeg','subject_4','session_1','processed-epo.fif')
epoch = mne.read_epochs(f_name, add_eeg_ref = False)


# Electrodes to calculate CDA
cda_left_id = [epoch.ch_names.index(ch) for ch in ['P5','P7','PO7','O1']]
cda_right_id = [epoch.ch_names.index(ch) for ch in ['P6','P8','PO8','O2']]


conditions = ['1_single','1_dual','2_single','2_dual']
for condition in conditions:
	left_hemi_left_cue = []
	left_hemi_right_cue = []
	right_hemi_left_cue = []
	right_hemi_right_cue = []
	for event in epoch.event_id.keys():
		if set(condition.split('_')).issubset(event.split('_')):
			if 'left' in event.split('_'):
				data_left = epoch[event].get_data()[:,cda_left_id,:]
				# baseline
				data_left = np.vstack([np.array(np.matrix(data_left[:,ch,:]) - np.matrix(data_left[:,ch,:102]).mean(axis = 1)) for ch in range(data_left.shape[1])])
				# filter
				data_left = butter_lowpass_filter(data_left, 5, 512, order = 6)

				data_right = epoch[event].get_data()[:,cda_right_id,:]
				# baseline
				data_right = np.vstack([np.array(np.matrix(data_right[:,ch,:]) - np.matrix(data_right[:,ch,:102]).mean(axis = 1)) for ch in range(data_right.shape[1])])
				# filter
				data_right = butter_lowpass_filter(data_left, 5, 512, order = 6)
			

				left_hemi_left_cue.append(data_left)
				right_hemi_left_cue.append(data_right)
			elif 'right' in event.split('_'):
				data_left = epoch[event].get_data()[:,cda_left_id,:]
				# baseline
				data_left = np.vstack([np.array(np.matrix(data_left[:,ch,:]) - np.matrix(data_left[:,ch,:102]).mean(axis = 1)) for ch in range(data_left.shape[1])])
				# filter
				data_left = butter_lowpass_filter(data_left, 5, 512, order = 6)

				data_right = epoch[event].get_data()[:,cda_right_id,:]
				# baseline
				data_right = np.vstack([np.array(np.matrix(data_right[:,ch,:]) - np.matrix(data_right[:,ch,:102]).mean(axis = 1)) for ch in range(data_right.shape[1])])
				# filter
				data_right = butter_lowpass_filter(data_left, 5, 512, order = 6)
			

				left_hemi_right_cue.append(data_left)
				right_hemi_right_cue.append(data_right)			

	left_hemi_left_cue = np.vstack(left_hemi_left_cue).mean(axis =0)			
	left_hemi_right_cue =  np.vstack(left_hemi_right_cue).mean(axis =0)
	right_hemi_left_cue =  np.vstack(right_hemi_left_cue).mean(axis =0)
	right_hemi_right_cue =  np.vstack(right_hemi_right_cue).mean(axis =0)

	f=plt.figure(figsize = (20,20))
	ax = f.add_subplot(2,1,1)
	plt.plot(np.arange(0,left_hemi_left_cue.size),left_hemi_left_cue,color = 'b', label = 'left cue')
	plt.plot(np.arange(0,left_hemi_right_cue.size),left_hemi_right_cue,color = 'r', label = 'right cue')
	plt.axvline(x = 102, color = 'black')	
	ax.legend(loc='upper right', shadow=True)
	plt.title('Left hemisphere')
	ax = f.add_subplot(2,1,2)
	plt.plot(np.arange(0,right_hemi_left_cue.size),right_hemi_left_cue,color = 'b', label = 'left cue')
	plt.plot(np.arange(0,right_hemi_right_cue.size),right_hemi_right_cue,color = 'r',label = 'right cue')
	plt.axvline(x = 102, color = 'black')
	ax.legend(loc='upper right', shadow=True)
	plt.title('Right hemisphere')
	plt.savefig(condition + '.pdf')

event_data = {}

conditions = ['1_single','1_dual','2_single','2_dual']

def butter_lowpass(cutoff, fs, order = 5):
	nyq = 0.5* fs
	normal_cutoff = cutoff/nyq
	b, a = butter(order, normal_cutoff, btype = 'low', analog = False)
	return b, a

def butter_lowpass_filter(data, cutoff, fs, order = 5):
	b,a = butter_lowpass(cutoff, fs, order = order)
	y = lfilter(b,a, data)	
	return y

