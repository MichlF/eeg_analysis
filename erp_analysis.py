"""
analyze EEG data

Created by Dirk van Moorselaar on 01-07-2015.
Copyright (c) 2015 DvM. All rights reserved.
"""

from scipy.signal import butter, lfilter, freqz
import pylab as pl
import os
import mne

# subject

all_subjects = {}

conditions = ['1_single','1_dual','2_single','2_dual']
for condition in conditions:
	all_subjects.update({condition:{}})
for subject in  ['subject_2','subject_3','subject_4']:
	

	# CDA conditions
	

	# CDA data
	epoch_info = []
	for session in ['session_1','session_2']:
		f_name = os.path.join('/Users','Dirk','Dropbox','Experiment_data','data','load_accessory','processed_eeg',subject,session,'processed-epo.fif')
		epoch_info.append(mne.read_epochs(f_name, add_eeg_ref = False))

	epoch_events = epoch_info[0].event_id

	# Electrodes to calculate CDA
	cda_hemi_left = [epoch_info[0].ch_names.index(ch) for ch in ['P5','P7','PO7','O1']]
	cda_hemi_right= [epoch_info[0].ch_names.index(ch) for ch in ['P6','P8','PO8','O2']]

	# per condition get data (Ipsi and Contra lateral)
	f=plt.figure(figsize = (20,20))
	for plot_id, condition in enumerate(conditions):
		all_subjects[condition].update({subject:[]})

		ipsi = []
		contra = []
		for event in epoch_events:
			if set(condition.split('_')).issubset(event.split('_')):
				if 'left' in event.split('_'):
					ipsi.append(np.vstack([epoch[event].get_data()[:,cda_hemi_left,:] for epoch in epoch_info]))
					contra.append(np.vstack([epoch[event].get_data()[:,cda_hemi_right,:] for epoch in epoch_info]))
				elif 'right' in event.split('_'):
					ipsi.append(np.vstack([epoch[event].get_data()[:,cda_hemi_right,:] for epoch in epoch_info]))
					contra.append(np.vstack([epoch[event].get_data()[:,cda_hemi_left,:] for epoch in epoch_info]))
				
		ipsi = np.vstack(ipsi)
		contra = np.vstack(contra)	
		
		# baseline correct(based on -0.4 tot 0; 205 samples) and average over all events and channels		
		ipsi = ipsi.reshape(ipsi.shape[0]*ipsi.shape[1], ipsi.shape[2])
		contra = contra.reshape(contra.shape[0]*contra.shape[1], contra.shape[2])

		ipsi_base = (np.array(np.matrix(ipsi) - np.matrix(ipsi[:,:205]).mean(axis = 1))).mean(axis = 0)
		contra_base = (np.array(np.matrix(contra) - np.matrix(contra[:,:205]).mean(axis = 1))).mean(axis = 0)
		
		# filter
		ipsi_filt = mne.filter.low_pass_filter(ipsi_base, 512, 5)
		contra_filt = mne.filter.low_pass_filter(contra_base, 512, 5)

		all_subjects[condition][subject] = [ipsi_filt, contra_filt] 

		time = ipsi_filt.shape[0]/epoch_info[0].info['sfreq']	* np.arange(ipsi_filt.shape[0],dtype = float)/float(ipsi_filt.shape[0]) - 0.4

		ax = f.add_subplot(2,2,plot_id + 1)
		plt.xlim(-0.4,ipsi_filt.shape[0]/epoch_info[0].info['sfreq']-0.4)
		plt.ylim(-3e-6,3e-6)
		#plt.xticks(np.arange(left_hemi_left_cue.shape[0]/epoch.info['sfreq']))	
		plt.plot(time,ipsi_filt,color = 'b', label = 'ipsi')
		plt.plot(time,contra_filt,color = 'r', label = 'contra')
		plt.axvline(x = 0, color = 'black')	
		plt.legend(loc='upper right', shadow=True)
		plt.title(condition)

	plt.savefig(subject + 'erp.pdf')


f = plt.figure(figsize = (20,20))
for plot_id, condition in enumerate(conditions):
	ax = f.add_subplot(2,2,plot_id + 1)
	plt.xlim(-0.4,ipsi_filt.shape[0]/epoch_info[0].info['sfreq']-0.4)
	plt.ylim(-3e-6,3e-6)
	ipsi = np.vstack([all_subjects[condition][key][0] for key in all_subjects[condition].keys()]).mean(axis = 0)
	contra = np.vstack([all_subjects[condition][key][1] for key in all_subjects[condition].keys()]).mean(axis = 0)
	plt.plot(time,ipsi,color = 'b', label = 'ipsi')
	plt.plot(time,contra,color = 'r', label = 'contra')
	plt.axvline(x = 0, color = 'black')	
	plt.legend(loc='upper right', shadow=True)
	plt.title(condition)

plt.savefig('subject_average_erp_0.1.pdf')

f=plt.figure(figsize = (8,8))
plot_data = []
for i, ch in enumerate(cda_hemi_left):
	ax = f.add_subplot(3,2,i + 1)
	plt.ylim(-2e-5,2e-5)
	if ch != 99:
		plot_data.append(data[1,i,:] - data[1,i,:205].mean())
		plt.plot(plot_data[i])
		plt.title(epoch.ch_names[ch])
	else:
		plot_data = np.vstack(plot_data).mean(axis = 0)	
		plt.plot(plot_data)
		plt.title('mean')
	
	plt.axvline(x = 205, color = 'black')
	








def butter_lowpass(cutoff, fs, order = 5):
	nyq = 0.5* fs
	normal_cutoff = cutoff/nyq
	b, a = butter(order, normal_cutoff, btype = 'low', analog = False)
	return b, a

def butter_lowpass_filter(data, cutoff, fs, order = 5):
	b,a = butter_lowpass(cutoff, fs, order = order)
	y = lfilter(b,a, data)	
	return y

