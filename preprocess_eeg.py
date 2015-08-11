"""
analyze EEG data

Created by Dirk van Moorselaar on 10-03-2015.
Copyright (c) 2015 DvM. All rights reserved.
"""

import mne
import os
import numpy as np
import scipy as sp
import logging
import matplotlib
import matplotlib.pyplot as plt
import math
import seaborn as sns
import itertools

try:
    from Tkinter import * #in python 2.x
except:
    from tkinter import * #in python 3.x

from matplotlib.collections import LineCollection
from mne.preprocessing.peak_finder import peak_finder
from IPython import embed as shell
from math import ceil, floor

# Define project root folder
this_raw_folder = os.path.join(os.getcwd(), 'EEG')
this_project_folder = os.path.join(os.getcwd())

class InteractiveGUI(Frame):
    """ 
    Creates the GUI for the EEG analysis menu 
    """
        
    def __init__(self, master):
        """ 
        Initialize the frame
        """
        
        Frame.__init__(self, master)
        self.grid()
        self.create_window()
        
    def combine_funcs(*funcs):
        """ 
        Combines several functions (or function calls) in one function
        Currently does not handle functions containing individual arguments (not tested)
        
        Arguments
        ------
        - any number of functions formatted as e.g.: (func1, func2, func3)
        
        Returns
        ------
        - combined function consisting of all input functions
        """
        
        def combined_func(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)
        return combined_func
        
    def create_window(self):
        """ 
        Creates the widgets
        """
        
        # predefine input vars
        self.ana_chosen = IntVar() 
        self.subjects_chosen = StringVar()
        self.runs_chosen = StringVar()
        self.combine_analyses = IntVar()
        self.run_preprocessing = IntVar()
        self.run_main = IntVar()
        self.ana_chosen.set(1)
        self.subjects_chosen.set('None')
        self.runs_chosen.set('None')
        self.combine_analyses.set(0) #set radio buttons empty at startup
        self.run_preprocessing.set(0)
        self.run_main.set(0)
        
        self.font_header = "Verdana 10 bold"
        self.font_regular = "Verdana 10"
        
        # create labels (text)
        Label(self,
              text = "Choose your analysis (all mandatory):", font = self.font_header
              ).grid(row = 0, column = 0, sticky = W, pady = 7)
        Label(self,
              text = "Analysis ID?", font = self.font_regular
              ). grid(row = 1, column = 0, sticky = W)
        Label(self,
              text = "Which subjects? (e.g. '2,4,5')", font = self.font_regular
              ).grid(row = 2, column = 0, sticky = W)
        Label(self,
              text = "Which runs/sessions? (e.g. '1,3')", font = self.font_regular
              ). grid(row = 3, column = 0, sticky = W)
        Label(self,
              text = "Select one (all optional, default 'No'):", font = self.font_header
              ). grid(row = 4, column = 0, sticky = W, pady = 7)
        Label(self,
              text = "Run runs/sessions combined?", font = self.font_regular
              ).grid(row = 5, column = 0, sticky = W)
        Label(self,
              text = "Run preprocessing?", font = self.font_regular
              ).grid(row = 6, column = 0, sticky = W)
        Label(self,
              text = "Run the main analysis?", font = self.font_regular
              ). grid(row = 7, column = 0, sticky = W)
        Label(self,
              text = "Current selection:", font = self.font_regular + ' italic'
              ).grid(row = 8, column = 0, sticky = W, pady = 10)
        
        # create text input windows
        self.ana_chosen = Entry(self)
        self.ana_chosen.grid(row = 1, column = 1, sticky = W, columnspan = 2)
        self.submit_button = Button(self,
                                    text = "Submit", font = self.font_regular, command = self.update_text
                                    ).grid(row = 1, column = 2, columnspan = 1, sticky = W+E, padx = 5)
        
        self.subjects_chosen = Entry(self)
        self.subjects_chosen.grid(row = 2, column = 1, sticky = W)
        self.submit_button = Button(self,
                                    text = "Submit", font = self.font_regular, command = self.update_text
                                    ).grid(row = 2, column = 2, columnspan = 2, sticky = W+E, padx = 5)

        self.runs_chosen = Entry(self)
        self.runs_chosen.grid(row = 3, column = 1, sticky = W)
        self.submit_button = Button(self,
                                    text = "Submit", font = self.font_regular, command = self.update_text
                                    ).grid(row = 3, column = 2, columnspan = 2, sticky = W+E, padx = 5)
        
        # create radiobuttons        
        Radiobutton(self,
                    text = "Yes", font = self.font_regular,
                    variable = self.combine_analyses, value = 1, command = self.update_text
                    ).grid(row = 5, column = 1, sticky = E)
        Radiobutton(self,
                    text = "No", font = self.font_regular,
                    variable = self.combine_analyses, value = 0, command = self.update_text
                    ).grid(row = 5, column = 2, sticky = W)
        Radiobutton(self,
                    text = "Yes", font = self.font_regular,
                    variable = self.run_preprocessing, value = 1, command = self.update_text
                    ).grid(row = 6, column = 1, sticky = E)
        Radiobutton(self,
                    text = "No", font = self.font_regular,
                    variable = self.run_preprocessing, value = 0, command = self.update_text
                    ).grid(row = 6, column = 2, sticky = W)
        Radiobutton(self,
                    text = "Yes", font = self.font_regular,
                    variable = self.run_main, value = 1, command = self.update_text
                    ).grid(row = 7, column = 1, sticky = E)
        Radiobutton(self,
                    text = "No", font = self.font_regular,
                    variable = self.run_main, value = 0, command = self.update_text
                    ).grid(row = 7, column = 2, sticky = W)

        # create text window
        self.text = Text(self, width = 60, height = 7, wrap = WORD)
        self.text.grid(row = 9, column = 0, columnspan = 3)
        
        # create some other buttons
        
        self.debug_mode = Button(self, text = "Debug Mode", font = self.font_regular, command = self.exit_window_2debug)
        self.debug_mode.grid(row = 10, column = 0, sticky = W, pady = 10, padx = 5)
        
        self.exit_button = Button(self, text = "Save and Start", font = self.font_regular)
        self.exit_button["command"] = self.exit_window
        self.exit_button.grid(row = 10, column = 2, sticky = E, pady = 10, padx = 5)
        
    def exit_window(self):
        """ 
        Exits the GUI to start analysis 
        """
        
        self.debug_mode = 0
        self.ana_number = self.ana_chosen.get()
        self.subjects = [int(n) for n in self.subjects_chosen.get().split(',')]
        self.run_number = [int(n) for n in self.runs_chosen.get().split(',')]
        self.combined_run = self.combine_analyses.get()
        self.step_preprocess = self.run_preprocessing.get()
        self.step_main = self.run_main.get()    
        
        self.master.destroy()
        
    def exit_window_2debug(self):
        """ 
        Exits the GUI to start analysis with debug parameters (defined in session batch script)
        """
        
        self.debug_mode = 1
        self.master.destroy()
        
    def update_text(self):
        """
        Updates the status window to the most recent choice input(s)
        Note: In the current version the status window remains blank until a choice/input is provided. This should be fixed at some point.
        """
        
        fb_ana_chosen = self.ana_chosen.get()
        fb_subjects_chosen = self.subjects_chosen.get()
        fb_runs_chosen = self.runs_chosen.get()
        fb_combine_analyses = 'not combined'
        fb_run_preprocessing = 'No preprocessing'
        fb_run_main = 'no main analysis'
        
        if self.combine_analyses.get() == 1:
            fb_combine_analyses = 'combined'
        if self.run_preprocessing.get() == 1:
            fb_run_preprocessing = 'Preprocessing'
        if self.run_main.get() == 1:
            fb_run_main = 'the main analysis'
            
        message = 'You have chosen '
        message += 'analysis ' + fb_ana_chosen + ' for subject(s) ' + fb_subjects_chosen + ' each with run(s)/session(s) ' + fb_runs_chosen + '.' + ' Their run(s)/session(s) will be ' + fb_combine_analyses + '.' + ' ' + fb_run_preprocessing + ' and ' + fb_run_main + ' will be executed.'
        
        self.text.delete(0.0, END)
        self.text.insert(0.0, message)

class InteractiveInInterpreter(object):
    """
    Allows user to fill in the analysis parameters inside the interpreter
    """
    
    def __init__(self, this_raw_folder):
        # Interactive details
        prompt = '---> '
        print ''
        print 'Please make sure you have your raw data files in {0}'.format(this_raw_folder)
        print ''
        print 'To start, please provide the following information...'
        print 'Is this debugging? (1 or 0)'
        self.debug = input(prompt)
        if self.debug == 1:
            print 'Which subjects? (e.g. "1,3,4")'
            self.subjects = [int(n) for n in raw_input(prompt).split(',')]
            print 'How many runs/sessions? (e.g. "1,2")'
            self.run_number = [int(n) for n in raw_input(prompt).split(',')]
            self.ana_number, self.combined_run, self.step_preprocess, self.step_main = 1, 1, 1, 1
        else:
            print 'What analysis? (int)'
            self.ana_number = input(prompt)
            print 'Which subjects? (e.g. "1,3,4")'
            self.subjects = [int(n) for n in raw_input(prompt).split(',')]
            print 'How many runs/sessions? (e.g. "1,2")'
            self.run_number = [int(n) for n in raw_input(prompt).split(',')]
            print 'Do you want to combine the runs/sessions? (1 or 0)'
            self.combined_run = input(prompt)
            print 'Run preprocessing? (1 or 0)'
            self.step_preprocess = input(prompt)
            print 'Run main analysis? (1 or 0)'
            self.step_main = input(prompt)
        
        # Print an overview info screen
        print ''
        print '-------------------------------------------------'
        print 'Initiate analysis of subjects {0}...'.format(self.subjects)
        print 'Steps included in analysis {0} with {1} runs:'.format(self.ana_number,self.run_number)
        print 'Preprocessing: {0}'.format(self.step_preprocess)
        print 'Main: {0}'.format(self.step_main)
        print '-------------------------------------------------'
        print ''
        
        if self.step_preprocess == 0 and self.step_main == 0:
            raise NameError('Please choose at least ONE analysis step')
        
    
class FolderStructure(object):
    """
    Creates the folder structure
    """
    
    def __init__(self, this_project_folder, subjects, ana_number, run_number, step_preprocess, step_main):
        try:
            os.makedirs(os.path.join(this_project_folder, 'first_level'))
        except OSError:
            print 'Could not create first level root folder. It may already exist.'
            
        try:
            os.makedirs(os.path.join(this_project_folder,'second_level','log'))
        except OSError:
            print 'Could not create second level root folder. It may already exist.'
            
        for subject_id in subjects:
            
            folders = ['data','anat','log']
            for folder in folders:
                try:
                    os.makedirs(os.path.join(this_project_folder,'first_level','subject_'+str(subject_id),folder))
                except OSError:
                    print 'Could not create subject folder(s)' 
                    
            # Remaining
            if step_preprocess == 1:
                for run in run_number: #for run in range(1,run_number + 1)
                    try:
                        os.makedirs(os.path.join(this_project_folder,'first_level','subject_'+str(subject_id),'ana_'+str(ana_number),'session_'+str(run),'preprocessed'))
                    except OSError:
                        print 'Could not create preprocess folder structure. It may already exist.'
                    try:
                        os.mkdir(os.path.join(this_project_folder,'first_level','subject_'+str(subject_id),'ana_'+str(ana_number),'session_'+str(run),'preprocessed','ICA'))
                        os.mkdir(os.path.join(this_project_folder,'first_level','subject_'+str(subject_id),'ana_'+str(ana_number),'session_'+str(run),'preprocessed','artifacts'))
                    except OSError:
                        print 'Could not created preprocessed folder(s). It may already exist.'                
            if step_main == 1: 
                for run in run_number:
                    try:
                        os.makedirs(os.path.join(this_project_folder,'first_level','subject_'+str(subject_id),'ana_'+str(ana_number),'session_'+str(run),'main'))
                    except OSError:
                        print 'Could not create main folder structure. It may already exist.'         
                    try:
                        os.mkdir(os.path.join(this_project_folder,'first_level','subject_'+str(subject_id),'ana_'+str(ana_number),'session_'+str(run),'main','ERP'))
                        os.mkdir(os.path.join(this_project_folder,'first_level','subject_'+str(subject_id),'ana_'+str(ana_number),'session_'+str(run),'main','time_freq'))
                        os.mkdir(os.path.join(this_project_folder,'first_level','subject_'+str(subject_id),'ana_'+str(ana_number),'session_'+str(run),'main','connect'))
                    except OSError:
                        print 'Could not create main folder(s). It may already exist.'
                        
    
class RawBDF(mne.io.edf.edf.RawEDF):
    '''
    Child originating from MNE built-in RawEDF, such that new methods can be added to this built in class
    '''

    def __init__(self,input_fname,subject_id, run_number, ana_number, montage = None, stim_channel  = -1, annot = None, annotmap = None, tal_channel = None, \
            hpts = None, preload = True, verbose = None):
        self.subject_id = subject_id
        self.run_number = run_number
        self.ana_number = ana_number
        super(RawBDF,self).__init__(input_fname, montage = montage, stim_channel  = stim_channel, annot = annot, annotmap = annotmap, tal_channel = tal_channel,\
             preload = preload, verbose = verbose)
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


        drop_channels = ['GSR1','GSR2','Erg1','Erg2','Resp','Plet','Temp']
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

        Returns
        - - - -
        self(object): raw object with changed channel names following biosemi 64 naming scheme (10 - 20 system)
        '''

        ch_names_dict = {
                'A1':'FP1', 'A2':'AF7','A3':'AF3', 'A4':'F1','A5':'F3', 'A6':'F5','A7':'F7', 'A8':'FT7','A9':'FC5', 'A10':'FC3',
                'A11':'FC1', 'A12':'C1','A13':'C3', 'A14':'C5','A15':'T7', 'A16':'TP7','A17':'CP5', 'A18':'CP3','A19':'CP1', 'A20':'P1',
                'A21':'P3', 'A22':'P5','A23':'P7', 'A24':'P9','A25':'PO7', 'A26':'PO3','A27':'O1', 'A28':'Iz','A29':'Oz', 'A30':'Poz','A31':'Pz', 
                'A32':'CPz','B1':'FPz', 'B2':'FP2','B3':'AF8', 'B4':'AF4','B5':'AFz', 'B6':'Fz','B7':'F2', 'B8':'F4','B9':'F6', 'B10':'F8',
                'B11':'FT8', 'B12':'FC6','B13':'FC4', 'B14':'FC2','B15':'FCz', 'B16':'Cz','B17':'C2', 'B18':'C4','B19':'C6', 'B20':'T8','B21':'TP8', 
                'B22':'CP6','B23':'CP4', 'B24':'CP2','B25':'P2', 'B26':'P4','B27':'P6', 'B28':'P8','B29':'P10', 'B30':'PO8','B31':'PO4', 'B32':'O2',
                'EXG1':'VEOG1','EXG2':'VEOG2','EXG3':'HEOG1','EXG4':'HEOG2','EXG7':'EOGBl','EXG8':'EOGEye',
                }
        
        channels = ch_names_dict.keys()
        new_channels = [ch_names_dict[key] for key in channels]

        for ch_ind, channel in enumerate(channels):
            self.ch_names[self.ch_names.index(channel)] = new_channels[ch_ind]
            self.info['chs'][self.ch_names.index(new_channels[ch_ind])]['ch_name'] = new_channels[ch_ind]        

    def reReference(self, ref_chans=['EXG5','EXG6'], vEOG = ['VEOG1','VEOG2','EOGBl'], hEOG = ['HEOG1','HEOG2','EOGEye']):
        '''
        Rereference raw data to reference channels. By default data is rereferenced to the mastoids.
        Also EOG data is rerefenced. Subtraction of VEOG and HEOG results in a blink channel and an eyemovement channel.
        After rereferencing redundant channels are removed  

        Arguments
        - - - - -
        self(object): RawBDF object 

        Returns
        - - - -
        
        self (object): Rereferenced raw eeg data
        '''

        # rereference all channels (EEG and EOG) to reference channels 
        logging.info('Data was rereferenced to channels ' + ', '.join(ref_chans))
        (self, ref_data) = mne.io.set_eeg_reference(self, ref_chans, copy=False)
        
        # rerefence EOG data (vertical and horizontal)
        v_id = [self.info['ch_names'].index(vert) for vert in vEOG]
        h_id = [self.info['ch_names'].index(vert) for hor in hEOG]

        self[v_id[2]][0][0] = self[v_id[0]][0] - self[v_id[1]][0]
        self[h_id[2]][0][0] = self[h_id[0]][0] - self[h_id[1]][0]


        # drop ref chans
        self.drop_channels(ref_chans)

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

        events = mne.find_events(self, stim_channel = stim_channel, consecutive = False) 
        # ADD UPDATE INCLUDE EVENT COUNT CHECK IN BEHAVIORAL LOG FILE
        for event_id, event in enumerate(events[:,2]):
            if (event in event_1) and events[event_id + 1, 2] in event_2:
                events[event_id,2] += 1000        
        self.event_list = events


class ProcessEpochs(mne.Epochs):
    '''
    Child originating from MNE built-in Epochs, such that new methods can be added to this built in class
    '''

    def __init__(self, raw_object, events, event_id, tmin, tmax, baseline, subject_id, run_number, ana_number, art_detect = False, trl_pad = 0, flt_pad = 0.5, art_pad = 0, box_car = 0.2, picks = None, preload = True, \
        decim = 1, on_missing = 'error', verbose = None, proj = False):
        self.subject_id = subject_id
        self.run_number = run_number
        self.ana_number = ana_number
        self.art_detect = art_detect
        self.flt_pad = flt_pad
        self.art_pad = art_pad
        self.trl_pad = trl_pad
        self.box_car = box_car
        self.epoch_len = tmax - tmin
        if art_detect:
            tmin -= (trl_pad + flt_pad)
            tmax += (trl_pad + flt_pad)
        super(ProcessEpochs,self).__init__(raw_object, events, event_id, tmin = tmin, tmax = tmax, baseline = baseline,\
            picks = picks, preload = preload, decim = decim, on_missing = on_missing, verbose = verbose, proj = proj)
        #logging.info

    def baselineEpoch(self, channel_id = 'all', epoch_id = None, baseline_period = False, baseline = (0,200)):
        '''
        Baseline correct Epochs. Function can either correct based on the average of the whole epoch or based on a specified period in ms.
        Functions combines all epochs into a single data array (channels by time).
        
        Arguments
        - - - - -
        self(object): Epochs object 
        channel_id (int | list | str): list of channel indices to apply baseline correction. Defaults to all 64 channels
        epoch_id (int | None): Index of epoch(s) to correct. Defaults to None (apply baseline correction to all epochs)
        baseline_period (bool): If False baseline correction is applied based on whole epoch.
        baseline (tuple): Start and end of baseline correction period in ms. Function assumes that an epoch starts at 0 ms. 


        Returns
        - - - -
        
        data_base (array): Array with baseline corrected data.
        '''

        if channel_id == 'all':
            channel_id = range(64)
        elif isinstance(channel_id, list) and isinstance(channel_id[0], str):
            channel_id = [self.info['ch_names'].index(ch) for ch in channel_id]    

        if epoch_id == None:
            data = np.hstack([self[epoch].get_data()[0,channel_id,:] for epoch in range(len(self))])    
            l_epoch = data.shape[1]/len(self)
        elif isinstance(epoch_id, int):
            data = self[epoch_id].get_data()[0,channel_id,:]
        
        if baseline_period:
            
            if baseline[0] != 0:
                start = int(self.info['sfreq']/(1000/baseline[0]))
            else:
                start = baseline[0]    
            end = int(self.info['sfreq']/(1000/baseline[1]))

            if isinstance(epoch_id, int):
                data_base = data - data[start:end].mean()
            else:
                data = data.reshape(nr_channels,len(self),l_epoch)
                data_base = np.vstack([np.array(np.matrix(data[ch,:,:]) - np.matrix(data[ch,:,start:end]).mean(axis = 1)).reshape(nr_epochs*l_epoch) for ch in channel_id])
        else:
            if isinstance(epoch_id, int):
                data_base = data - data.mean()
            else:
                data_base = np.array(np.hstack([np.matrix(data[:,i*l_epoch:i*l_epoch + l_epoch]) - np.matrix(data[:,i*l_epoch:i*l_epoch + l_epoch]).mean(axis=1) for i in range(len(self))]))

        return data_base


    def artifactDetection(self, z_cutoff = 4, nr_channels = 64, band_pass = [110,140], plt_range = 1e-04, plot = True):
        '''
        Detect artifacts based on FieldTrip's automatic artifact detection. Artifacts are detected in three steps:
            1. Filtering the data
            2. Z-transforming the filtered data and normalize it over channels
            3. Threshold the accumulated z-score

        Arguments
        - - - - -
        self(object): Epochs object 
        z_cutoff (int): Value that is added to difference between median and min value of accumulated z-score to obtain z-threshold
        nr_channels (int): Number of channels used for artifact detection. Defualt (64) uses only EEG channels.
        band_pass (list): Low and High frequency cutoff for band_pass filter
        plot (bool): If True save detection plots (overview of z scores across epochs, raw signal of channel with highest z score, z distributions, raw signal of all electrodes)


        Returns
        - - - -
        
        self.marked_epochs (data): Adds a list of marked epochs to Epoch object
        '''

        epoch_data = []
        
        for ep in range(len(self)):

        # STEP 1: select data and filter data 
            data = self[ep].get_data()[0,range(nr_channels),:]
            
            if ep == 0:
                id_epoch = np.ones(data.shape[1], dtype = bool)
                id_epoch[:int((self.flt_pad*self.info['sfreq']))] = False
                id_epoch[int((-self.flt_pad*self.info['sfreq'])):] = False
            data = mne.filter.band_pass_filter(data, self.info['sfreq'],band_pass[0], band_pass[1], method = 'iir', iir_params = dict(order=9, ftype='butter')) # CHECK FILTER DIRECTION, compare to FIELDTRIP SETTINGS 
            data = np.abs(sp.signal.hilbert(data))
            # box_car smoothing
            data = self.boxSmoothing(data)

            # remove filter padding
            data = data[:,id_epoch]
            epoch_data.append(data)

        # STEP 2: Z-transform data
        epoch_comb = np.hstack(epoch_data)
        avg_data = epoch_comb.mean(axis = 1).reshape(-1,1)
        std_data = epoch_comb.std(axis = 1).reshape(-1,1)
        z_data = [(epoch - avg_data)/std_data  for epoch in epoch_data] 
        
        # STEP 3 threshold z-score per epoch
        z_accumel = np.hstack(z_data).sum(axis = 0)/math.sqrt(nr_channels)
        z_accumel_ep = [np.array(z.sum(axis = 0)/math.sqrt(nr_channels)) for z in z_data]
        z_thresh = np.median(z_accumel) + abs(z_accumel.min() - np.median(z_accumel)) + z_cutoff

        # Loop over all epochs to plot results
        bad_epoch = []

        for ep in range(len(self)):

            ch_id = np.where(z_data[ep] == z_data[ep].max())[0][0]

            # check if epoch contains artifact
            if (z_accumel_ep[ep] > z_thresh).sum() > 0:
                bad_epoch.append(ep)

            if plot:
                data_2_plot = self[ep].get_data()[0][ch_id,id_epoch]
                z_2_plot = z_accumel_ep[ep]    
                if ep == 0:
                    id_trial = np.ones(data_2_plot.size, dtype = bool)
                    id_trial[:int((self.trl_pad*self.info['sfreq']))] = False
                    id_trial[int((-self.trl_pad*self.info['sfreq'])):] = False

                plt.figure(figsize = (40,20))
                with sns.axes_style('dark'):
                    
                    matplotlib.rcParams.update({'font.size':70})

                    ax = plt.subplot(121, xlabel = 'samples', ylabel = 'z_value', xlim = (0,z_accumel.size), ylim = (-20,40))
                    plt.plot(np.arange(0,z_accumel.size),z_accumel,color = 'b')
                    plt.plot(np.arange(0,z_accumel.size)[ep*id_trial.size:ep*id_trial.size + id_trial.size],z_2_plot,color = 'r')
                    plt.plot(np.arange(0,z_accumel.size),np.ma.masked_less(z_accumel, z_thresh),color = 'r')
                    ax.axhline(z_thresh, color = 'r', ls = '--')

                    start, end = (self.tmin + self.flt_pad, self.tmax - self.flt_pad)

                    ax = plt.subplot(222, xlabel = 'Time (s)', ylabel = 'EEG signal (V)', title = 'Epoch ' + str(ep) + ', channel ' + self.ch_names[ch_id], xlim = (start,end))
                    plt.plot(np.arange(start,end,abs(start-end)/data_2_plot.size),data_2_plot,color = '#95B8BA')    
                    plt.plot(np.arange(start,end,abs(start-end)/data_2_plot.size)[id_trial],data_2_plot[id_trial],color = 'b')    

                    ax = plt.subplot(224, xlabel = 'Time (s)', ylabel = 'z_value', xlim = (start,end), ylim = (-20,40))
                    plt.plot(np.arange(start,end,abs(start-end)/data_2_plot.size),z_2_plot,color = '#95B8BA')
                    plt.plot(np.arange(start,end,abs(start-end)/data_2_plot.size)[id_trial],z_2_plot[id_trial],color = 'b')
                    plt.plot(np.arange(start,end,abs(start-end)/data_2_plot.size),np.ma.masked_less(z_2_plot, z_thresh),color = 'r')
                    ax.axhline(z_thresh, color = 'r', ls = '--')    

                plt.savefig(os.path.join(this_project_folder,'first_level','subject_'+str(self.subject_id),'ana_'+str(self.ana_number), \
                    'session_'+str(self.run_number),'preprocessed','artifacts','epoch_' + str(ep) + '.pdf'))
                plt.close()            

        np.savetxt(os.path.join(this_project_folder,'first_level','subject_'+str(self.subject_id),'ana_'+str(self.ana_number), \
                    'session_'+str(self.run_number),'preprocessed','artifacts','marked_epochs.txt'), \
                np.array(bad_epoch, dtype = int), fmt = '%u')    
                

    def boxSmoothing(self, data):
        '''
        doc string boxSmoothing
        '''

        pad = int(round(self.box_car*self.info['sfreq']))
        if pad % 2 == 0:
            # the kernel should have an odd number of samples
            pad += 1
            kernel = np.ones(pad)/pad
        pad = int(ceil(pad/2))
        pre_pad = int(min([pad, floor(data.shape[1])/2.0]))
        edge_left = data[:,:pre_pad].mean(axis = 1)
        edge_right = data[:,-pre_pad:].mean(axis = 1)
        data = np.concatenate((np.tile(edge_left.reshape(data.shape[0],1),pre_pad),data, np.tile(edge_right.reshape(data.shape[0],1),pre_pad)), axis = 1)
        data_smooth = sp.signal.convolve2d(data,kernel.reshape(1,kernel.shape[0]),'same')
        data = data_smooth[:,pad:(data_smooth.shape[1]-pad)]

        return data

    def filterEpoch(self,signal, sampl_freq = 512, low_pass = 110, high_pass = 140):
        '''
        doc string filterEpoch
        '''

        b, a = sp.signal.butter(3,[low_pass/2.0/sampl_freq, high_pass/2.0/sampl_freq], btype = 'band')
        return sp.signal.filtfilt(b,a,signal)

    def dropMarkedEpochs(self):
        '''
        Drop epochs as marked by the noise detection (artifactDetection and detectEyeMovements) functions. Uses MNE built in drop_epochs function.
        For extra info see doc string of mne.drop_epochs

        Arguments
        - - - - -
        self(object): Epochs object 

        Returns
        - - - -
        
        self(object): Epochs object without marked epochs
        '''    

        drop_epochs = np.loadtxt(os.path.join(this_project_folder,'first_level','subject_'+str(self.subject_id),'ana_'+str(self.ana_number), \
                    'session_'+str(self.run_number),'preprocessed','artifacts','marked_epochs.txt'),dtype = int)
        self.drop_epochs(drop_epochs, reason = 'User marked')


    def correctArtifactICA(self, n_components = 50, nr_channels = 64, EOG = ['VEOG1','VEOG2'], max_comp = 4):
        '''
        docstring: CHECK WHETHER HEOG NEEDS TO BE INCLUDED!!!!!!!
        '''

        # initiate ICA
        ica = mne.preprocessing.ICA(n_components = n_components, method = 'infomax').fit(self, picks = range(nr_channels))
        
        eog_index, scores = ica.find_bads_eog(self, 'EOGBl')
        show_picks = np.abs(scores).argsort()[::-1][:5]
        ica.plot_scores(scores, exclude = eog_index, show = False).savefig(os.path.join(this_project_folder,'first_level','subject_'+str(self.subject_id),'ana_'+str(self.ana_number), \
                    'session_'+str(self.run_number),'preprocessed','ICA','scores_.pdf'))
        plt.close()
        ica.plot_sources(self,show_picks, exclude = eog_index, title = 'Sources related to EOG artifacts (red)', show= False).savefig(os.path.join(this_project_folder,'first_level','subject_'+str(self.subject_id),'ana_'+str(self.ana_number), \
                    'session_'+str(self.run_number),'preprocessed','ICA','sources_.pdf'))
        plt.close()
        layout = mne.layouts.read_layout(os.path.join(this_project_folder,'subject_layout_64_std.lout'))
        ica.plot_components(range(20), colorbar = True, layout = layout, ch_type = 'eeg', show = False).savefig(os.path.join(this_project_folder,'first_level','subject_'+str(self.subject_id),'ana_'+str(self.ana_number), \
                    'session_'+str(self.run_number),'preprocessed','ICA','eog_comp_all.pdf'))
        plt.close()
        

        if eog_index != []: 
            ica.plot_components(eog_index[: max_comp], colorbar = True, layout = layout, ch_type = 'eeg', show = False).savefig(os.path.join(this_project_folder,'first_level','subject_'+str(self.subject_id),'ana_'+str(self.ana_number), \
                    'session_'+str(self.run_number),'preprocessed','ICA','eog_comp_sel.pdf'))
        else:
            ica.plot_components(show_picks[:4], colorbar = True, layout = layout, ch_type = 'eeg', show = False).savefig(os.path.join(this_project_folder,'first_level','subject_'+str(self.subject_id),'ana_'+str(self.ana_number), \
                    'session_'+str(self.run_number),'preprocessed','ICA','eog_comp_sel.pdf'))    
        plt.close()    

        #eog_events = self.detectBlinks()
        
        # select eeg data (1 sec) around blink events 
        #all_data = np.hstack([self[epoch].get_data()[0,:nr_electrodes,:] for epoch in range(len(self))]) 
        #blinks_eeg =np.array([all_data[:,id -self.info['sfreq']/2.0:id+self.info['sfreq']/2.0] for id in eog_events])

        # initiate ICA
        #layout = mne.layouts.read_layout(os.path.join('/Users','Dirk','Dropbox','eeg_analysis','subject_layout_64_std.lout'))
        #ica = mne.preprocessing.ICA(n_components = n_components)
        #ica.fit(self, picks = range(nr_electrodes), decim = 3)

        # select components to remove
        #for eog_ch in EOG:
        #    # detect EOG by correlation
        #    eog_index, scores = ica.find_bads_eog(self, eog_ch)

        #    ica.plot_scores(scores, exclude = eog_index, show = False).savefig(os.path.join('/Users','Dirk','Dropbox','Experiment_data','data',\
        #        'load_accessory','processed_eeg','subject_' + str(self.subject_id), 'session_' + str(self.run_number), 'figs','ica', 'scores_' + eog_ch + '.pdf'))
        #    plt.close()

        #    show_picks = np.abs(scores).argsort()[::-1][:5]
            
        #    ica.plot_sources(self,show_picks, exclude = eog_index, title = 'Sources related to EOG artifacts (red)', show= False).savefig(os.path.join('/Users','Dirk','Dropbox','Experiment_data','data',\
        #        'load_accessory','processed_eeg','subject_' + str(self.subject_id), 'session_' + str(self.run_number), 'figs','ica', 'sources_' + eog_ch + '.pdf'))
        #    plt.close()

        #    if eog_index != []: 
        #        ica.plot_components(eog_index[: max_comp], colorbar = True, layout = layout, ch_type = 'eeg', show = False).savefig(os.path.join('/Users','Dirk','Dropbox','Experiment_data','data',\
        #        'load_accessory','processed_eeg','subject_' + str(self.subject_id), 'session_' + str(self.run_number), 'figs', 'ica','eog_comp_' + eog_ch + '.pdf'))
        #    else:
        #        ica.plot_components(show_picks[0], colorbar = True, layout = layout, ch_type = 'eeg', show = False).savefig(os.path.join('/Users','Dirk','Dropbox','Experiment_data','data',\
        #        'load_accessory','processed_eeg','subject_' + str(self.subject_id), 'session_' + str(self.run_number), 'figs', 'ica','eog_comp_empty' + eog_ch + '.pdf'))    
        #    plt.close()    
                
        #    ica.exclude += eog_index[:max_comp]
        #    logging.info('Component {0} excluded with score of {1}'.format(eog_index[:max_comp],scores[eog_index][:max_comp]))

        # remove selected components
        #self = ica.apply(self)  
    
        # select eeg data (1 sec) around blink events (after ica)
        #all_data = np.hstack([self[epoch].get_data()[0,:nr_electrodes,:] for epoch in range(len(self))]) 
        #blinks_eeg_ica =np.array([all_data[:,id -self.info['sfreq']/2.0:id+self.info['sfreq']/2.0] for id in eog_events])

        # plot ICA effect
        #for blink_id in range(blinks_eeg_ica.shape[0] + 1):
        #    f=plt.figure(figsize = (40,40))
                    
        #    with sns.axes_style('dark'):

        #        if blink_id == 0:
        #            data = blinks_eeg.mean(axis = 0).T
        #            data_ica = blinks_eeg_ica.mean(axis = 0).T
        #            epoch = 'mean_epoch'
        #        else:
        #            data = blinks_eeg[blink_id-1].T
        #            data_ica = blinks_eeg_ica[blink_id-1].T
        #            epoch = 'epoch_' + str(blink_id -1)
#
        #        ax = f.add_subplot(211)    
        #        self.plotEEG(data,ax)
        #        ax = f.add_subplot(212)    
        #        self.plotEEG(data_ica,ax)
        #
        #    plt.savefig(os.path.join('/Users','Dirk','Dropbox','Experiment_data','data','load_accessory','processed_eeg',\
        #            'subject_' + str(self.subject_id), 'session_' + str(self.run_number), 'figs','ica', 'ica_effect' + epoch + '.pdf'))
        #    plt.close()

        # crop epochs to control for filter padding and save epoched data
        #self.crop(self.tmin + self.filter_padding, self.tmax - self.filter_padding)        
        #self.save(os.path.join('/Users','Dirk','Dropbox','Experiment_data','data','load_accessory','processed_eeg','subject_' + str(self.subject_id), \
        #'session_' + str(self.run_number), 'processed-epo.fif'))


    def plotEEG(self,data, ax, plt_range = 1e-4, nr_channels = 64, fill = False, x_min = None, x_max = None):
        '''
        docstring
        '''

        time = data.shape[0]/self.info['sfreq']    * np.arange(data.shape[0],dtype = float)/float(data.shape[0])
        plt.xlim(0,data.shape[0]/self.info['sfreq'])
        plt.xticks(np.arange(data.shape[0]/self.info['sfreq']))    
        
        dr = (plt_range)*0.35
        y_min, y_max = -plt_range, (data.shape[1] - 1) * dr + plt_range
        plt.ylim(y_min,y_max)

        segs = []
        ticklocs = []
        for ch in range(nr_channels):
            segs.append(np.hstack((time[:,np.newaxis], data[:,ch,np.newaxis])))
            ticklocs.append(ch*dr)

        offsets = np.zeros((nr_channels,2), dtype = float)
        offsets[:,1] = ticklocs

        lines = LineCollection(segs, offsets = offsets, transOffset = None)

        ax.add_collection(lines)
        ax.set_yticks(ticklocs)
        ax.set_yticklabels(self.info['ch_names'][:nr_channels])
        plt.xlabel('Time (s) ')
        plt.ylabel('Electrode channels')
        if fill:
            plt.fill_between(time[np.where(np.logical_and(time>=x_min, time<=x_max))[0]],y_min, y_max,color = 'red', alpha = 0.1)

    def detectEyeMovements(self, channels = ['EOGEye'], threshold = 25e-6, t_window = [-0.1,1], window = 0.1, step = 0.05, plot = True):
        '''
        Detect eye movements by marking step like activity that is greater than a given threshold. Based on pop_artstep.m from the ERPLAB Tolbox. 
        
        Arguments
        - - - - -
        self(object): Epochs object 
        channels (list): Channels used to detect eye movements
        threshold (int): Threshold set to check whether voltage shift is specified as an eye movement
        t_window (list): Time window in seconds used to scan for eye movements
        window (float): Window in seconds used to calculate average activity
        step (float): Step size (sliding window approach) 
        plot (bool): If True, plots epoch data (EOG Channel) of detected blink

        Returns
        - - - -
        
        eye_epochs (data): Saves list of epochs containing eye movements
        '''

        if self.tmin > t_window[0] or self.tmax < t_window[1]:
            raise ValueError('Selected time points not within epoch selection')

        ch_id = np.array([self.info['ch_names'].index(ch) for ch in channels])
        data = [self[ep].get_data()[0,ch_id,:] for ep in range(len(self))]

        # create sliding window (adjust for sampling frequency)
        window = int(self.info['sfreq']/(1.0/window))
        step = int(self.info['sfreq']/(1.0/step))
        start_id, end_id = int(self.info['sfreq']/(1.0/abs(self.tmin - t_window[0]))), int(self.info['sfreq']/(1.0/abs(self.tmin - t_window[1])))

        sl_wind = [[i, i + window] for i in range(start_id, end_id, step)]    
        if sl_wind[-1][1] > end_id:
            sl_wind[-1][1] = end_id

        # per epoch apply sliding window     
        eye_epoch = []    
        for ind, ep_data in enumerate(data):
            for i in range(len(sl_wind) - 1):
                act_1 = ep_data.squeeze()[sl_wind[i][0]:sl_wind[i][1]].mean()
                act_2 = ep_data.squeeze()[sl_wind[i + 1][0]:sl_wind[i + 1][1]].mean()
                if abs(act_1 - act_2) > threshold:
                    eye_epoch.append(ind)
                    if plot:
                        #data_2_plot = self.baselineEpoch(channel_id = ch_id, epoch_id = ind).squeeze()
                        data_2_plot = ep_data    
                        plt.xlabel('Time(s)')
                        plt.ylabel('EOG signal (V')
                        #plt.ylim(-threshold*2, threshold*2)    
                        plt.plot(np.arange(self.tmin,self.tmax,abs(self.tmin-self.tmax)/data[id].size),data_2_plot,color = '#95B8BA')
                        plt.plot(np.arange(self.tmin,self.tmax,abs(self.tmin-self.tmax)/data[id].size)[start_id:end_id],data_2_plot[start_id:end_id],color = 'b')
                        plt.savefig(os.path.join(this_project_folder,'first_level','subject_'+str(self.subject_id),'ana_'+str(self.ana_number), \
                    'session_'+str(self.run_number),'preprocessed','artifacts','eye_epoch_' + str(ind) + '.pdf'))
                        plt.close()
                    break    

        np.savetxt(os.path.join(this_project_folder,'first_level','subject_'+str(self.subject_id),'ana_'+str(self.ana_number), \
                    'session_'+str(self.run_number),'preprocessed','artifacts','eye_epochs.txt'), \
            np.array(eye_epoch, dtype = int), fmt = '%u')            
    
    def detectBlinks(self,band_pass = [1,10], ch_name = ['VEOG1','VEOG2']):
        '''
        docstring
        '''

        # loop over EOG channels
        eog_id = [self.info['ch_names'].index(ch) for ch in ch_name]
        eog = np.vstack([np.hstack([self[epoch].get_data()[0,id,:] for epoch in range(len(self))]) for id in eog_id])
        len_epoch = eog.shape[1]/len(self)
        eog_base = np.array(np.hstack([np.matrix(eog[:,i*len_epoch:i*len_epoch + len_epoch]) - np.matrix(eog[:,i*len_epoch:i*len_epoch + len_epoch]).mean(axis=1) \
                for i in range(len(self))]))

        # filtering to remove dc offset to dissociate between blinks and saccades
        fmax = np.minimum(45, self.info['sfreq']/2.0 -0.75)
        filt_eog = self.filterEpoch(eog_base, self.info['sfreq'], low_pass = 2, high_pass = fmax)
        id_max = np.argmax(np.sqrt(np.sum(filt_eog ** 2,axis = 1)))
        
        # easier to detect peaks with filtering
        filt_eog = self.filterEpoch(eog[id_max], self.info['sfreq'], low_pass = band_pass[0], high_pass = band_pass[1])

        # detecting eog blinks
        temp = filt_eog - filt_eog.mean()
        if np.abs(np.max(temp)) > np.abs(np.min(temp)):
            eog_events, _ = peak_finder(filt_eog, extrema = 1)
        else:
            eog_events, _ = peak_finder(filt_eog, extrema = -1)

        # plot blinks
        blinks = np.array([eog[id_max,id-self.info['sfreq']/2.0:id+self.info['sfreq']/2.0] for id in eog_events])


        plt.plot(blinks.T)
        plt.axvline(x = self.info['sfreq']/2.0, color = 'r')
        plt.ylabel(ch_name[id_max])
        plt.ylabel('Time (ms)')
        plt.savefig(os.path.join(this_project_folder,'first_level','subject_'+str(self.subject_id),'ana_'+str(self.ana_number), \
                    'session_'+str(self.run_number),'preprocessed','ICA','detected_blinks.pdf'))
        plt.close()

        # return time index blinks    
        return eog_events







class RawEpochs(mne.Epochs):
    '''
    Child originating from MNE built-in Epochs, such that new methods can be added to this built in class
    '''    


    def __init__(self, raw_object, events, event_id, tmin, tmax, baseline = (None,0), picks = None, preload = True, \
            decim = 1, on_missing = 'error', verbose = None):
        super(RawEpochs,self).__init__(raw_object, events, event_id, tmin = tmin, tmax = tmax, baseline = baseline,\
            picks = picks, preload = preload, decim = decim, on_missing = on_missing, verbose = verbose)
        #logging.info

    def markHighFreqEpochs(self, sampl_freq = 512, window = 100, step = 50, threshold = 120e-6, channels = 5):
        '''
        Flag all epochs that contain high frequency noise as defined by different parameters
        '''

        # loop over all epochs
        bad_epochs = []
        all_data = np.hstack([EO[epoch].get_data()[0] for epoch in range(len(EO))])
        
        all_peaks = [np.array(detectPeaks(all_data[ch], valley = False)) for ch in range(64)]
        all_valleys = [np.array(detectPeaks(all_data[ch], valley = True)) for ch in range(64)]

        for i in range(64):
            if len(all_peaks[i]) < len(all_valleys[i]):
                all_valleys[i] = all_valleys[i][:len(all_peaks[i])]
            elif len(all_valleys[i]) < len(all_peaks[i]):
                all_peaks[i] = all_peaks[i][:len(all_valleys[i])]

        mean_amp = [np.mean(all_data[i][all_peaks[i]] - all_data[i][all_valleys[i]]) for i in range(len(all_peaks))]        
        std_amp = [np.std(all_data[i][all_peaks[i]] - all_data[i][all_valleys[i]]) for i in range(len(all_peaks))]

        for epoch in range(len(EO)):    
            
            data = EO[epoch].get_data()[0]
            
            # create sliding window
            sl_window = [(i,i + window) for i in range(0,data.shape[1] - window, step)]
            if sl_window[-1][-1] < data.shape[1] - 1:
                sl_window.append((sl_window[-1][0] + 50, data.shape[1] - 1))

            peak_info = []
            ps_info = []
            mark_epoch = False    
            for i, (start, stop) in enumerate(sl_window):
                
                # for each time window calculate peak to peak amplitude and power spectrum
                peak2peak = [data[ch,start:stop].max() - data[ch,start:stop].min() for ch in range(64)] # peak to peak amplitude per electrode
                for j in range(len(mean_amp)):
                    if peak2peak[j] > mean_amp[j] + 20*std_amp[j]:
                        mark_epoch = True

                peak2peak_sorted = np.array(sorted((f,e) for e, f in enumerate(peak2peak)))
                
                ps = [(np.abs(np.fft.fft(data[ch,start:stop]))**2)[:int(window/2)] for ch in range(64)] # power spectrum per electrode

                peak_info.append(peak2peak_sorted)
                ps_info.append(ps)

            if mark_epoch:
                bad_epochs.append(epoch)

                windows2plot = np.array([peak_info[i][:,0].max() for i in range(len(peak_info))]).argsort()[-2:]    
                channels2plot = np.array(peak_info[windows2plot[-1]],dtype = int)[-4:,1]

                freqs_w = np.fft.fftfreq(window, 1.0/sampl_freq)[:int(window/2)]
                freqs_e = np.fft.fftfreq(data[0].size, 1.0/sampl_freq)[:int(data[0].size/2)]
            
                plot_index_epoch = [1,3,9,11]
                plot_index_window = [(3,4),(7,8),(19,20),(23,24)]

                f=plt.figure(figsize = (20,20))

                for ch_index in range(len(channels2plot)):
                    
                    # plot epoch
                    data2plot = data[channels2plot[ch_index]]
                    ax = f.add_subplot(4,4,plot_index_epoch[ch_index])
                    plt.plot(np.arange(0,data2plot.size),data2plot,color = 'g')
                    # show mean amplitude in plot
                    plt.plot(np.arange(0,data2plot.size),[mean_amp[channels2plot[ch_index]] + 10*std_amp[channels2plot[ch_index]]]*data2plot.size,color ='black')
                    plt.plot(np.arange(0,data2plot.size),[-mean_amp[channels2plot[ch_index]] - 10*std_amp[channels2plot[ch_index]]]*data2plot.size,color ='black')
                    plt.fill_between(np.arange(0,data2plot.size),-mean_amp[channels2plot[ch_index]],mean_amp[channels2plot[ch_index]],alpha = 0.7, color = 'g')

                    plt.title('Channel ' + EO.ch_names[ch_index])
                    plt.xlim(0,data2plot.size)
                    if ch_index in [0,2]:
                        plt.ylabel('EEG signal')
                    else:
                        plt.yticks([])
                    plt.ylim(-threshold/2,threshold/2)

                    # plot power spectrum
                    ax = f.add_subplot(4,4,plot_index_epoch[ch_index] + 4)
                    ps2plot = (np.abs(np.fft.fft(data[ch,:]))**2)[:int(data[0].size/2)]
                    plt.plot(freqs_e, ps2plot,color = 'r')
                    if ch_index in [0,2]:
                        plt.ylabel('Power Spectrum')
                    else:
                        plt.yticks([])

                    # plot window of epoch
                    for win_ind, win in enumerate(windows2plot[::-1]):
                        
                        data2plot = data[channels2plot[ch_index],sl_window[win][0]:sl_window[win][1]]
                        ax = f.add_subplot(4,8,plot_index_window[ch_index][win_ind])
                        plt.plot(np.arange(sl_window[win][0],sl_window[win][1]),data2plot,color = 'b')
                        plt.fill_between(np.arange(sl_window[win][0],sl_window[win][1]),-mean_amp[channels2plot[ch_index]],mean_amp[channels2plot[ch_index]],alpha = 0.7, color = 'b')

                        plt.yticks([])
                        plt.xlim(sl_window[win][0],sl_window[win][1])
                        plt.ylim(-threshold/2,threshold/2)
                    # plot window of power spectrum        
                        ps2plot = ps_info[win_ind][ch_index]
                        ax = f.add_subplot(4,8,plot_index_window[ch_index][win_ind] + 8)
                        plt.plot(freqs_w, ps2plot,color = 'y')
                        plt.yticks([])
                        
                plt.savefig(data_folder + os.path.join('/load_accessory/eeg/marked_epoch' + str(epoch) + '.pdf'))
                plt.close()        

            for peak in peak_info:
                if peak[-1][0] > threshold:

                    # plot data channels with highest amplitudes
                    f=plt.figure(figsize = (40,40))

                    for plots in range(len(peak_info)):
                        for ch in range(channels):
                            # plot signal for separate time bins
                            ind_channel = peak_info[plots][-ch-1][1]
                            data2plot = data[ind_channel,sl_window[plots][0]:sl_window[plots][1]]
                            

                            ax = f.add_subplot(len(peak_info),channels,ch + 1 + plots*channels)
                            plt.plot(np.arange(sl_window[plots][0],sl_window[plots][1]),data2plot,color = 'g')
                            plt.title('Channel ' + EO.ch_names[ind_channel])
                            
                            #plt.xlabel('time (ms)')
                            if ch == 0:
                                plt.ylabel('EEG signal')
                            else:
                                plt.yticks([])    
                            plt.ylim(-150e-6,150e-6)
                            plt.xlim(sl_window[plots][0],sl_window[plots][1])
                            #plt.xlim(sl_window[plots][0],sl_window[plots][1])
                            #
                            #plt.text(x = 0.37, y = 0.95, s = 'p2p = ' + str(round(peak2peak_sorted[-ch-1][0],5)),verticalalignment='center',fontsize=20, transform = ax.transAxes, bbox = {'alpha' :0.3, 'facecolor': 'grey' })
                            #plt.axis('off')

                            # plot frequency plots for all time bins
                            #ax = f.add_subplot(len(peak_info) + 1,channels*2,plot)
                            #plt.plot(freqs, ps_info[plots][peak_info[plots][-ch-1][1]],color = 'r')
                            
                            #plt.fill_between(freqs[np.where(freqs>50)],0,np.max(ps_info[plots][peak_info[plots][-ch-1][1]]),alpha = 0.2, color = 'grey')
                            #plt.xlabel('freq (Hz)')
                            #if ch == 0:
                            #    plt.ylabel('Power Spectrum')
                            #else:
                            #    plt.yticks([])
                            #plt.ylim(0,np.max(ps[peak2peak_sorted[-ch-1][1]]))
                            #plt.xlim(freqs[0],freqs[-1])    
                                
                    plt.savefig(data_folder + os.path.join('/load_accessory/eeg/fig' + str(epoch) + 'win_' + str(i) + '.pdf'))
                    plt.close()                
                    break
                        
            
            return unique(bad_epochs)            
    



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
            _plot(signal, ind,mph, mpd, threshold, edge, valley, ax)

        return ind 

    def _plot(x, ind, mph = None, mpd = 1, threshold = 0, edge =' rising', valley = False, ax = None):
        """Plot results of the detect_peaks function, see its help."""
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is not available.')
        else:
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(8, 4))

            plt.plot(x, 'b', lw=1)
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












# and define events
#events = mne.find_events(raw_eeg, stim_channel = 'STI 014')


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



    def eogEpochs(self, nr_electrodes = 64):
        '''
        docstring for EOGEpochs
        '''

        eog_epoch = mne.preprocessing.create_eog_epochs(self,'VEOG1',999,picks = range(nr_electrodes),tmin = -0.5,tmax = 0.5,baseline = (None,None)) 
        eog_epoch.save(os.path.join(this_project_folder,'first_level','subject_'+str(self.subject_id),'ana_'+str(self.ana_number), \
                    'session_'+str(self.run_number),'preprocessed','artifacts','eog-epo.fif'))
  




