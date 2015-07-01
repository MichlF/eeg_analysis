"""
analyze EEG data

Created by Dirk van Moorselaar on 10-03-2015.
Copyright (c) 2015 DvM. All rights reserved.
"""

from preprocess_eeg import *
from IPython import embed as shell
import os 


this_raw_folder = os.path.join('/Users','Dirk','Dropbox','Experiment_data','data','load_accessory','raw_eeg') 
this_project_folder = os.path.join('/Users','Dirk','Dropbox','Experiment_data','data','load_accessory','processed_eeg')
subjects = 4

def runWholeSession(ra, Ea,session):

	###############
	#PREPROCESSING

	session.dropEmptyChannels()
	session.renameChannel()
	session.changeEventCodes(event_1 = [100,109,110,119,200,209,210,219],
							event_2	= [101,102,103,104,105,106,107,108,
										111,112,113,114,115,116,117,118,
										201,202,203,204,205,206,207,208,
										211,212,213,214,215,216,217,218])
	session.reReference()
	session.filter(l_freq = 0.5, h_freq = None, h_trans_bandwidth = 0.1)

	session = ProcessEpochs(session,session.event_list,Ea[0]['event_id_mem'],Ea[1]['timing_mem'][0],Ea[1]['timing_mem'][1],session.subject_id,session.session_id,None)
	session.detectEyeMovements()
	session.artifactDetection(z_threshold = 20, plot = True) # CHECK METHOD TO SPECIFY Z VALUE
	session.dropMarkedEpochs()
	session.correctArtifactICA()

if __name__ == '__main__':

	try:
		os.mkdir(this_project_folder)
	except OSError:
		pass

	for subject_id in range(2,subjects + 1):	
		EEG_run_array = [
				{'session' : 1, 'raw_data_path': os.path.join(this_raw_folder,'subject' + str(subject_id) + '_session_1.bdf')},
				{'session' : 2, 'raw_data_path': os.path.join(this_raw_folder,'subject' + str(subject_id) + '_session_2.bdf')},
						]

		EEG_ERP_array = [
				{'event_id_mem' : {'1_match_left_single' :100,'1_match_right_single': 109,'1_neutral_left_single': 110,'1_neutral_right_single': 119,'2_match_left_single': 200,
								'2_match_right_single': 209,'2_neutral_left_single': 210,'2_neutral_right_single': 219,'1_match_left_dual': 1100,'1_match_right_dual': 1109,
								'1_neutral_left_dual': 1110,'1_neutral_right_dual': 1119,'2_match_left_dual': 1200,'2_match_right_dual': 1209,'2_neutral_left_dual': 1210,
								'2_neutral_right_dual': 1219}, },

				{'timing_mem':[-0.2,1.2,None]},	# tmin, tmax, baseline	CHECK WHETHER BASELINE CORRECTION IS NECESSARY!!!!!!!!!!!!		
				#{'timing_mem':[0.8,1.9,None]},

				{'event_id_search': {'1_match_up_left': 101,'1_match_up_right': 102,'1_match_down_left': 103,'1_match_down_right': 104,'1_match_left_up': 105,'1_match_left_down': 106,
									'1_match_right_up': 107,'1_match_right_down': 108,'1_neutral_up_left': 111,'1_neutral_up_right': 112,'1_neutral_down_left': 113,
									'1_neutral_down_right': 114,'1_neutral_left_up': 115,'1_neutral_left_down': 116,'1_neutral_right_up': 117,'1_neutral_right_down': 118,
									'2_match_up_left': 201,'2_match_up_right': 202,'2_match_down_left': 203,'2_match_down_right': 204,'2_match_left_up': 205,'2_match_left_down': 206,
									'2_match_right_up': 207,'2_match_right_down': 208,'2_neutral_up_left': 211,'2_neutral_up_right': 212,'2_neutral_down_left': 213,'2_neutral_down_right': 214,
									'2_neutral_left_up': 215,'2_neutral_left_down': 216,'2_neutral_right_up': 217,'2_neutral_right_down': 218}},						]		

		for session in EEG_run_array:						
			EEG_session = RawBDF(session['raw_data_path'],subject_id,session['session'])

			runWholeSession(EEG_run_array, EEG_ERP_array, EEG_session)	

