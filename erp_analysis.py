import scipy.signal
import pylab as pl

# Electrodes to calculate CDA
cda_electr_left = ['P5','P7','PO7','O1']
cda_electr_right = ['P6','P8','PO8','O2']
cda_electr_id_left = [[i for i, ch_name in enumerate(epochs.ch_names) if ch_name == electr][0] for electr in cda_electr_left]
cda_electr_id_right = [[i for i, ch_name in enumerate(epochs.ch_names) if ch_name == electr][0] for electr in cda_electr_right]

event_data = {}

conditions = ['1_single','1_dual','2_single','2_dual']


for condition in conditions:
	data_ipsi = []
	data_contra = []
	for event in event_id_mem.keys():
		if set(condition.split('_')).issubset(event.split('_')):
			if 'left' in event.split('_'):
				data_ipsi.append(epochs[event].get_data()[:,cda_electr_id_left,:])
				data_contra.append(epochs[event].get_data()[:,cda_electr_id_right,:])
			elif 'right' in event.split('_'):	
				data_ipsi.append(epochs[event].get_data()[:,cda_electr_id_right,:])
				data_contra.append(epochs[event].get_data()[:,cda_electr_id_left,:])



	event_data.update({condition: {'ipsi': np.vstack(data_ipsi), 'contra': np.vstack(data_contra) }})		


f=pl.figure(figsize = (16,7))
for i, cond in enumerate(conditions):
	subplot = 221 + i
	ax = f.add_subplot(subplot)
	for j, side in enumerate(['ipsi','contra']):
		data = event_data[cond][side].mean(axis = (0,1))
		data = scipy.signal.savgol_filter(data,75,2)
		pl.plot(data, color = ['b','r'][j])
	ax.set_title(cond)	


