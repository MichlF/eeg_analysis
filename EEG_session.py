"""
analyze EEG data

Created by Dirk van Moorselaar on 10-03-2015.
Copyright (c) 2015 DvM. All rights reserved.
"""

from preprocess_eeg import *
from IPython import embed as shell
import os

#################
### Functions ###
#################

def runWholeSession(ra,Ea,session):    

    ###############
    #PREPROCESSING
    session.dropEmptyChannels()
    session.renameChannel()
    session.changeEventCodes(event_1 = [100,109,110,119,200,209,210,219], # ADJUST FUNCTION PER EXPERIMENT
                            event_2    = [101,102,103,104,105,106,107,108,
                                        111,112,113,114,115,116,117,118,
                                        201,202,203,204,205,206,207,208,
                                        211,212,213,214,215,216,217,218])
    
    # Before rereferencing and filtering, apply semi-automatic rejection of EMG-contaminated epochs
    temp_session = ProcessEpochs(session, session.event_list, Ea[0]['event_id_mem'], Ea[1]['timing_mem'][0], Ea[1]['timing_mem'][1], \
                    None, session.subject_id,session.run_number, session.ana_number, art_detect = True, trl_pad = 0.5, flt_pad = 0.5, art_pad = 0) 
    temp_session.artifactDetection(z_cutoff = 4, plot = False)

    # Continue preprocessing with raw data
    session.reReference() 
    session.filter(l_freq = 0.1, h_freq = None, filter_length = 3000, l_trans_bandwidth = 0.095) 
    session = ProcessEpochs(session, session.event_list, Ea[0]['event_id_mem'], Ea[1]['timing_mem'][0], Ea[1]['timing_mem'][1], \
                    (None, None), session.subject_id, session.run_number, session.ana_number, art_detect = False) 
    #session.detectEyeMovements()
    session.dropMarkedEpochs()
    
    # Removing eye-blinks with Independent Component Analysis
    session.correctArtifactICA()

###############################################
### Define project root and raw data folder ###
###############################################

this_raw_folder = os.path.join(os.getcwd(), 'EEG')
this_project_folder = os.path.join(os.getcwd())

########################
### Fill in the data ###
########################

### -> Interactive menu
root = Tk()
root.title("EEG analysis menu")
analysis_parameters = InteractiveGUI(root)
root.mainloop()

### -> Alternative: Interactive text in the interpreter
#analysis_parameters = InteractiveInInterpreter(this_raw_folder)

### -> With debug mode on (For hard coding the parameters)
if analysis_parameters.debug_mode == 1:
    analysis_parameters.subjects = [1]
    analysis_parameters.ana_number = 1
    analysis_parameters.run_number = [1]
    analysis_parameters.step_preprocess = 1
    analysis_parameters.step_main = 1

###############################
### Create folder structure ###
###############################

FolderStructure(this_project_folder, analysis_parameters.subjects, analysis_parameters.ana_number, analysis_parameters.run_number, analysis_parameters.step_preprocess, analysis_parameters.step_main)

#####################
### Read triggers ###
#####################
         
if __name__ == '__main__':

    for subject_id in analysis_parameters.subjects:
        if len(analysis_parameters.run_number) == 1:
            EEG_run_array = [
                {'run_number' : analysis_parameters.run_number, 'ana_number' : analysis_parameters.ana_number, 'raw_data_path': os.path.join(this_raw_folder,'subject' + str(subject_id) + '.bdf')}
                    ]
        else:
            EEG_run_array = []
            for runner in analysis_parameters.run_number:
                EEG_run_array.append({'run_number' : runner, 'ana_number' : analysis_parameters.ana_number, 'raw_data_path': os.path.join(this_raw_folder,'subject' + str(subject_id) + '_session_' + str(runner) + '.bdf')})

        EEG_ERP_array = [
                {'event_id_mem' : {'1_match_left_single' :100,'1_match_right_single': 109,'1_neutral_left_single': 110,'1_neutral_right_single': 119,'2_match_left_single': 200,
                                '2_match_right_single': 209,'2_neutral_left_single': 210,'2_neutral_right_single': 219,'1_match_left_dual': 1100,'1_match_right_dual': 1109,
                                '1_neutral_left_dual': 1110,'1_neutral_right_dual': 1119,'2_match_left_dual': 1200,'2_match_right_dual': 1209,'2_neutral_left_dual': 1210,
                                '2_neutral_right_dual': 1219}, },

                {'timing_mem':[-0.2,1.2,(None,None)]},

                {'event_id_search': {'1_match_up_left': 101,'1_match_up_right': 102,'1_match_down_left': 103,'1_match_down_right': 104,'1_match_left_up': 105,'1_match_left_down': 106,
                                    '1_match_right_up': 107,'1_match_right_down': 108,'1_neutral_up_left': 111,'1_neutral_up_right': 112,'1_neutral_down_left': 113,
                                    '1_neutral_down_right': 114,'1_neutral_left_up': 115,'1_neutral_left_down': 116,'1_neutral_right_up': 117,'1_neutral_right_down': 118,
                                    '2_match_up_left': 201,'2_match_up_right': 202,'2_match_down_left': 203,'2_match_down_right': 204,'2_match_left_up': 205,'2_match_left_down': 206,
                                    '2_match_right_up': 207,'2_match_right_down': 208,'2_neutral_up_left': 211,'2_neutral_up_right': 212,'2_neutral_down_left': 213,'2_neutral_down_right': 214,
                                    '2_neutral_left_up': 215,'2_neutral_left_down': 216,'2_neutral_right_up': 217,'2_neutral_right_down': 218}},                        ]        

        
        if analysis_parameters.combined_run == 1:                        
            EEG_session_comb = mne.concatenate_raws([RawBDF(run_number['raw_data_path'], subject_id, run_number['run_number'], run_number['ana_number']) for run_number in EEG_run_array])
            runWholeSession(EEG_run_array, EEG_ERP_array, EEG_session_comb)    

        else:            
            for run_number in EEG_run_array:                        
                EEG_session = RawBDF(run_number['raw_data_path'], subject_id, run_number['run_number'], run_number['ana_number'])
                runWholeSession(EEG_run_array, EEG_ERP_array, EEG_session)    