"""
analyze EEG data

Created by Dirk van Moorselaar on 10-03-2015.
Copyright (c) 2015 DvM. All rights reserved.
"""

import mne
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from IPython import embed as shell


data_folder = os.path.join('/Users','Dirk','Dropbox','Experiment_data','data') 
file_name = data_folder + '/load_accessory/eeg/subject1_session_1.bdf'

# step 1
event_1 = [100,109,200,209,110,119,210,219]
event_2 = [101,102,103,104,105,106,107,108,111,112,113,114,115,116,117,118,201,202,203,204,205,206,207,208,211,212,213,214,215,216,217,218]

RO = RawBDF(file_name)
RO.dropEmptyChannels()
RO.renameChannel()
RO.reReference()
RO.filter(l_freq = 0.5, h_freq = None, h_trans_bandwidth = 0.1)
RO.changeEventCodes(event_1, event_2)

#step 2
event_id_mem = {'1_match_left_single' :100,
				'1_match_right_single': 109,
				'1_neutral_left_single': 110,
				'1_neutral_right_single': 119,
				'2_match_left_single': 200,
				'2_match_right_single': 209,
				'2_neutral_left_single': 210,
				'2_neutral_right_single': 219,
				'1_match_left_dual': 1100,
				'1_match_right_dual': 1109,
				'1_neutral_left_dual': 1110,
				'1_neutral_right_dual': 1119,
				'2_match_left_dual': 1200,
				'2_match_right_dual': 1209,
				'2_neutral_left_dual': 1210,
				'2_neutral_right_dual': 1219,		
				}
tmin = 0.8
tmax = 1.9
baseline = (0.8,1.0)				
EO = RawEpochs(RO, RO.event_list, event_id_mem, tmin, tmax, baseline)


class RawBDF(mne.io.edf.edf.RawEDF):
	'''
	Child originating from MNE built-in RawEDF, such that new methods can be added to this built in class
	'''

	def __init__(self,input_fname,n_eeg = 64, stim_channel  = -1, annot = None, annotmap = None, tal_channel = None, \
			hpts = None, preload = True, verbose = None):
		super(RawBDF,self).__init__(input_fname, stim_channel  = stim_channel, annot = annot, annotmap = annotmap, tal_channel = tal_channel,\
		 	hpts = hpts, preload = preload, verbose = verbose)
		#logging.info('rawBDF instance was created for subject {0}'.format(input_fname[-8:]))


	def dropEmptyChannels(self, channels_to_remove = ['C','D']):
		'''
		Remove channels with no signal.  

		Arguments
		- - - - -
		self(object): RawBDF object 

		Returns
		- - - -
		
		self (object): RawBDF with reduced number of channels
		'''

		if len(self.ch_names) > 73:
			drop_channels = []
		for channel in channels_to_remove:
			drop_channels += [channel + str(i) for i in range(1,33)]
		
			self.drop_channels(drop_channels)

	def renameChannel(self):
		'''
		Change channel labels from A, B etc naming scheme to standard naming conventions. At the same time changes the name of EOG electrodes (assumes that
		an EXG naming scheme)

		Arguments
		- - - - - 
		raw (object): contains rereferenced eeg data 
		channels (list of strings): name of cnannels to be renamed
		new_names (list of strings): new names for the channels specified by channels argument

		'''

		ch_names_dict = {
				'A1':'FP1', 'A2':'AF7','A3':'AF3', 'A4':'F1','A5':'F3', 'A6':'F5','A7':'F7', 'A8':'FT7','A9':'FC5', 'A10':'FC3',
				'A11':'FC1', 'A12':'C1','A13':'C3', 'A14':'C5','A15':'T7', 'A16':'TP7','A17':'CP5', 'A18':'CP3','A19':'CP1', 'A20':'P1',
				'A21':'P3', 'A22':'P5','A23':'P7', 'A24':'P9','A25':'PO7', 'A26':'PO3','A27':'O1', 'A28':'Iz','A29':'Oz', 'A30':'Poz','A31':'Pz', 
				'A32':'CPz','B1':'FPz', 'B2':'FP2','B3':'AF8', 'B4':'AF4','B5':'AFz', 'B6':'Fz','B7':'F2', 'B8':'F4','B9':'F6', 'B10':'F8',
				'B11':'FT8', 'B12':'FC6','B13':'FC4', 'B14':'FC2','B15':'FCz', 'B16':'Cz','B17':'C2', 'B18':'C4','B19':'C6', 'B20':'T8','B21':'TP8', 
				'B22':'CP6','B23':'CP4', 'B24':'CP2','B25':'P2', 'B26':'P4','B27':'P6', 'B28':'P8','B29':'P10', 'B30':'PO8','B31':'PO4', 'B32':'O2',
				'EXG1':'VEOG1','EXG2':'VEOG2','EXG3':'HEOG3','EXG4':'HEOG4','EXG7':'EOGBl','EXG8':'EOGEye',
				}
		
		channels = ch_names_dict.keys()
		new_channels = [ch_names_dict[key] for key in channels]

		for ch_ind, channel in enumerate(channels):
			self.ch_names[self.ch_names.index(channel)] = new_channels[ch_ind]
			self.info['chs'][self.ch_names.index(new_channels[ch_ind])]['ch_name'] = new_channels[ch_ind]		

	def reReference(self, ref_chans=['EXG5','EXG6']):
		'''
		Rereference raw data to reference channels. By default data is rereferenced to the mastoids  

		Arguments
		- - - - -
		self(object): RawBDF object 

		Returns
		- - - -
		
		self (object): Rereferenced raw eeg data
		'''

		logging.info('Data was rereferenced to channels ' + ', '.join(ref_chans))
		(self, ref_data) = mne.io.set_eeg_reference(self, ref_chans, copy=False)
        
		for ch_ind, channel in enumerate(ref_chans):
			self.info['chs'][self.ch_names.index(channel)]['kind'] = 502
        
		self.info['bads'] += ref_chans 

	def changeEventCodes(self, event_1 = [], event_2 = [], stim_channel = 'STI 014'):
		'''
		Change event codes. Function changes specific event codes of event_1 when immediately followed by event_2  

		Arguments
		- - - - -
		self(object): RawBDF object 
		event_1 (list): list of event codes of first relevant event in trial
		event_2 (list): list of event codes of second relevant event in trial

		Returns
		- - - -
		
		self.event_list (data): Adds event_list to RawBDF object
		'''

		events = mne.find_events(self, stim_channel = stim_channel)


		for event_id, event in enumerate(events[:,2]):
			if (event in event_1) and events[event_id + 1, 2] in event_2:
				events[event_id,2] += 1000

		self.event_list = events		


class RawEpochs(mne.Epochs):
	'''
	Child originating from MNE built-in Epochs, such that new methods can be added to this built in class
	'''	


	def __init__(self, raw_object, events, event_id, tmin, tmax, baseline = (None,0), picks = None, preload = True, \
			decim = 1, on_missing = 'error', verbose = None):
		super(RawEpochs,self).__init__(raw_object, events, event_id, tmin = tmin, tmax = tmax, baseline = baseline,\
			picks = picks, preload = preload, decim = decim, on_missing = on_missing, verbose = verbose)
		#logging.info

	def markHighFreqEpochs(self, sampl_freq = 1.95, window = 100, step = 50, threshold = 100e-5):
		'''
		Flag all epochs that contain high frequency noise as defined by different parameters
		'''

		# loop over all epochs
		bad_epochs = []

		for epoch in range(len(EO)):	
			
			data = EO[epoch].get_data()
			
			# create sliding window
			sl_window = [(i,i + window) for i in range(0,data.shape[2] - window, step)]
			if sl_window[-1][-1] < data.shape[2] - 1:
				sl_window.append((sl_window[-1][0] + 50, data.shape[2] - 1))

			for start, stop in sl_window:

				peak2peak = [abs(data[0,ch,start:stop].min()) + abs(data[0,ch,start:stop].max()) for ch in range(64)] # peak to peak amplitude per electrode

				if max(peak2peak) > threshold:
					bad_epochs.append(epoch)
			






    
	def detectPeaks(signal, mph = None, mpd = 1, threshold = 0, edge = 'rising', kpsh = False, valley = False, show = False, ax = None):
    	'''
    	Detect peaks in data based on their amplitude and other features.
    	Function copied from 'Marcos Duarte, https://github.com/demotu/BMC'

    	Arguments
    	- - - - - 
    	signal (np.array): 1D_array data
    	mph (int): detect peaks that are greater 
    	mpd (int): optional (default = 1), detect peaks that are at least separated by minimum peak distance (in
        number of data).
    	threshold (int): optional (default = 0), detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    	edge (string): {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    	kpsh (boolean): optional (default = False), keep peaks with same height even if they are closer than `mpd`.
    	valley (bool): optional (default = False), if True (1), detect valleys (local minima) instead of peaks.
    	
    	show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
   		 ax : a matplotlib.axes.Axes instance, optional (default = None).

    	Returns
    	- - - -

    	index (np.array): 1D_array indices of the peaks (valleys)
    	'''   
	
		signal = np.atleast_1d(signal).astype('float64')
		if signal.size < 3:
			return np.array([], dtype=int)
		if valley:
			signal = -signal
	    # find indices of all peaks
		dx = signal[1:] - signal[:-1]
	    # handle NaN's
		indnan = np.where(np.isnan(signal))[0]
		if indnan.size:
			signal[indnan] = np.inf
			dx[np.where(np.isnan(dx))[0]] = np.inf
		ine, ire, ife = np.array([[], [], []], dtype=int)
		if not edge:
			ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
		else:
			if edge.lower() in ['rising', 'both']:
				ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
			if edge.lower() in ['falling', 'both']:
				ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
		ind = np.unique(np.hstack((ine, ire, ife)))
		# handle NaN's
		if ind.size and indnan.size:
	        # NaN's and values close to NaN's cannot be peaks
			ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
	    # first and last values of x cannot be peaks
		if ind.size and ind[0] == 0:
			ind = ind[1:]
		if ind.size and ind[-1] == signal.size-1:
			ind = ind[:-1]
	    # remove peaks < minimum peak height
		if ind.size and mph is not None:
			ind = ind[signal[ind] >= mph]
	    # remove peaks - neighbors < threshold
		if ind.size and threshold > 0:
			dx = np.min(np.vstack([signal[ind]-signal[ind-1], signal[ind]-signal[ind+1]]), axis=0)
			ind = np.delete(ind, np.where(dx < threshold)[0])
	    # detect small peaks closer than minimum peak distance
		if ind.size and mpd > 1:
			ind = ind[np.argsort(signal[ind])][::-1]  # sort ind by peak height
			idel = np.zeros(ind.size, dtype=bool)
			for i in range(ind.size):
				if not idel[i]:
	                # keep peaks with the same height if kpsh is True
					idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
					& (signal[ind[i]] > x[ind] if kpsh else True)
					idel[i] = 0  # Keep current peak
	        # remove the small peaks and sort back the indices by their occurrence
			ind = np.sort(ind[~idel])

		if show:
			if indnan.size:
				signal[indnan] = np.nan
			if valley:
				signal = -signal
			_plot(signal, mph, mpd, threshold, edge, valley, ax, ind)

		return ind 



	def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
		"""Plot results of the detect_peaks function, see its help."""
		
		try:
			import matplotlib.pyplot as plt
		except ImportError:
			print('matplotlib is not available.')
		else:
			if ax is None:
				_, ax = plt.subplots(1, 1, figsize=(8, 4))

			ax.plot(x, 'b', lw=1)
			if ind.size:
				label = 'valley' if valley else 'peak'
				label = label + 's' if ind.size > 1 else label
				ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
					label='%d %s' % (ind.size, label))
				ax.legend(loc='best', framealpha=.5, numpoints=1)
			ax.set_xlim(-.02*x.size, x.size*1.02-1)
			ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
			yrange = ymax - ymin if ymax > ymin else 1
			ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
			ax.set_xlabel('Data #', fontsize=14)
			ax.set_ylabel('Amplitude', fontsize=14)
			mode = 'Valley detection' if valley else 'Peak detection'
			ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"% (mode, str(mph), mpd, str(threshold), edge))
			# plt.grid()
			plt.show()




    def CorrectArtifactICA(raw, n_components = 50, picks = None, EOG = ['VEOG1','VEOG2','HEOG3','HEOG4'], max_comp = 1):
		'''
		docstring
		'''

		# initiate ICA
		ica = mne.preprocessing.ICA(n_components = n_components)
		ica.fit(self, picks = picks, decim = 3)
		
		# select components to remove
		for eog_ch in EOG:
			eog_index, scores = ica.find_bads_eog(self, eog_ch)
			ica.exclude += eog_index[: max_comp]
			logging.info('Component {0} excluded with score of {1}'.format(eog_index[:max_comp],scores[eog_index][:max_comp]))

		return ica.apply(self)    




## input data




## step 1: read in raw data and change channel names
data_folder = os.path.join('/Users','Dirk','Dropbox','Experiment_data','data') 
file_name = data_folder + '/load_accessory/eeg/subject1_session_1.bdf'

raw_eeg = mne.io.read_raw_edf(input_fname = file_name, preload = True)







# and define events
events = mne.find_events(raw_eeg, stim_channel = 'STI 014')
mem_events = [100,109,200,209,110,119,210,219]
search_events = [101,102,103,104,105,106,107,108,111,112,113,114,115,116,117,118,201,202,203,204,205,206,207,208,211,212,213,214,215,216,217,218]

for event_id, event in enumerate(events[:,2]):
	if (event in mem_events) and events[event_id + 1, 2] in search_events:
		events[event_id,2] += 1000

# step 2: rereference to linked mastoids
#raw_eeg = setEEGReference(raw = raw_eeg)
#raw_eeg, _ = mne.io.set_eeg_reference(raw_eeg, ['EXG5','EXG6'], copy=False)

# and rereference EOG data
#raw_eeg = setEOGReference(raw = raw_eeg)

# step 4: filter data
#raw_eeg.filter(l_freq = 0.5, h_freq = None, h_trans_bandwidth = 0.1) # now only EEG channels are filtered (needs to be checked)

# step 5: split data into different epochs and apply baseline correction


# specify events
event_id_mem = {'1_match_left_single' :100,
				'1_match_right_single': 109,
				'1_neutral_left_single': 110,
				'1_neutral_right_single': 119,
				'2_match_left_single': 200,
				'2_match_right_single': 209,
				'2_neutral_left_single': 210,
				'2_neutral_right_single': 219,
				'1_match_left_dual': 1100,
				'1_match_right_dual': 1109,
				'1_neutral_left_dual': 1110,
				'1_neutral_right_dual': 1119,
				'2_match_left_dual': 1200,
				'2_match_right_dual': 1209,
				'2_neutral_left_dual': 1210,
				'2_neutral_right_dual': 1219,		
				}

event_id_search =  {'1_match_up_left': 101,
					'1_match_up_right': 102,
					'1_match_down_left': 103,
					'1_match_down_right': 104,
					'1_match_left_up': 105,
					'1_match_left_down': 106,
					'1_match_right_up': 107,
					'1_match_right_down': 108,
					'1_neutral_up_left': 111,
					'1_neutral_up_right': 112,
					'1_neutral_down_left': 113,
					'1_neutral_down_right': 114,
					'1_neutral_left_up': 115,
					'1_neutral_left_down': 116,
					'1_neutral_right_up': 117,
					'1_neutral_right_down': 118,
					'2_match_up_left': 201,
					'2_match_up_right': 202,
					'2_match_down_left': 203,
					'2_match_down_right': 204,
					'2_match_left_up': 205,
					'2_match_left_down': 206,
					'2_match_right_up': 207,
					'2_match_right_down': 208,
					'2_neutral_up_left': 211,
					'2_neutral_up_right': 212,
					'2_neutral_down_left': 213,
					'2_neutral_down_right': 214,
					'2_neutral_left_up': 215,
					'2_neutral_left_down': 216,
					'2_neutral_right_up': 217,
					'2_neutral_right_down': 218,
					}


reject = dict(eeg=100e-5)
epochs = mne.Epochs(raw = raw_eeg, events = events, event_id = event_id_mem, tmin = 0.8, tmax = 1.9, baseline = (0.8,1.0), preload = True)#reject = reject, reject_tmin = 0.8, reject_tmax = 1.9, preload = True) # picks should be integers

# step 5: 
#ICA for artifact correction
epochs = CorrectArtifactICA(epochs)
#raw_eeg = CorrectArtifactICA(raw_eeg)


#ica = mne.preprocessing.ICA(n_components = 50)
#ica.fit(raw_eeg, picks = picks_eeg, decim = 3)
#ica_eeg = ica.apply(raw_eeg)

def renameChannel(raw,channels, new_names):
	'''
	Change channel label to new_name

	Arguments
	- - - - - 
	raw (object): contains rereferenced eeg data 
	channels (list of strings): name of cnannels to be renamed
	new_names (list of strings): new names for the channels specified by channels argument

	'''
	if type(channels) not in  [list,tuple]:
	    raise TypeError('Channels argument is supposed to be a list or tuple, not {0}. See doc'.format(type(channels)))
	if type(new_names) not in  [list,tuple]:
	    raise TypeError('new_name argument is supposed to be a list or tuple, not {0}. See doc'.format(type(channels)))      

	for chI, channel in enumerate(channels):
	    raw.ch_names[raw.ch_names.index(channel)] = new_names[chI]
	    raw.info['chs'][raw.ch_names.index(new_names[chI])]['ch_name'] = new_names[chI]

def setEOGReference (raw, ref_channels_blink = ['VEOG1','VEOG2'], ref_channels_eye = ['HEOG3','HEOG4'], dif_channel = ['VEOG','HEOG']):
	"""
	function to reference eog Channels

	Needs to be adjusted!!!!!!!!!!!!!!!!!!!!!
	
	Arguments
	- - - - - -

	raw (object): contains rereferenced eeg data  
	ref_channels_blink: (list of strings): the name (or names) of the eog channels that detect eye blinks
	ref_channels_eye: (list of strings): the name (or names) of the eog channels that detect eye movements
	dif_channel (list of strings): the name (or names) of the rereferenced difference channels

	Returns
	- - - - - 
	raw (object): the raw eeg with referenced eog channels. Raw eeg now contains EOG electrodes and HEOG and VEOG difference channels

	"""

	# find the indices to rereference the electrodes
	ref_index_blink = [raw.ch_names.index(i) for i in ref_channels_blink]
	ref_index_eye = [raw.ch_names.index(i) for i in ref_channels_eye]

	# create differnce channels
	for ch in dif_channel:
		raw.ch_names.append(ch)

	# compute blink and eye movements
	eye_data = np.vstack((raw._data[ref_index_blink[0]] - raw._data[ref_index_blink[1]],raw._data[ref_index_eye[0]] - raw._data[ref_index_eye[1]]))

	# append data to BDF file
	raw._data = np.vstack((raw._data,eye_data))

	return raw 




  




