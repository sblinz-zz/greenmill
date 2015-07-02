from sklearn import linear_model
import pandas as pd
import numpy as np
import pickle
import os
import etl

def get_std_scores_and_stats(df):
	df2 = pd.DataFrame(index = df.index, columns = df.columns)
	mins = []
	ranges = []

	for i in range(len(etl.features)):
		mins.append(df[etl.features[i]].min())
		ranges.append(df[etl.features[i]].max() - mins[-1])
		df2[etl.features[i]] = (df[etl.features[i]] - mins[-1])/ranges[-1]
    
	return df2, mins, ranges

def train_and_save_logreg(train_df, features):
		
	model = linear_model.LogisticRegression(class_weight='auto')

	#Build model and verify
	exo_vars, mins, ranges = get_std_scores_and_stats(train_df[etl.features])
	endo_var = train_df['outcome']
	model.fit(exo_vars, endo_var)

	#Save the model
	#Save the mins and ranges objects for standardizing submitted data on the site
	stats = [mins, ranges]
	pickle.dump(stats, open(os.path.join(os.path.dirname(__file__), 'stats.p'), 'wb'))
	pickle.dump(model, open(os.path.join(os.path.dirname(__file__), 'logreg.p'), 'wb'))

"""
Compute Input Vector from individual stage durations
def ComputeFeatureVectorFromWebInput(stages, num_stages):
	
	At the end of etl::GetFeatureDFFromTransformedClassDF() we rearrange the feature colummns in the following order:
		['dT_1','dT_f','dT_t',
		'dT_1/dT_f','dT_1/dT_t','dT_f/dT_t',
		'dT_min','dT_avg','dT_max',
		'rk_f','prog_f','n_stages']
	
	
	ok_stages = []
	for stage in stages:
		try:
			stage = float(stage)
			ok_stages.append(stage)
		except ValueError:
			pass

		try:
			num_stages = int(num_stages)
		except ValueError:
			num_stages = len(ok_stages)

	#Single candidate demo assumes rk_1 = 1
	dT_1 = ok_stages[0]
	dT_f = ok_stages[-1]
	dT_t = sum(ok_stages)
	dT_1_f = dT_1/dT_f
	dT_1_t = dT_1/dT_t
	dT_f_t = dT_f/dT_t
	dT_min = min(ok_stages)
	dT_avg = dT_t/len(ok_stages)
	dT_max = max(ok_stages)
	rk_f = len(ok_stages)
	prog_f = rk_f / num_stages

	return [dT_1, dT_f, dT_t, dT_1_f, dT_1_t, dT_f_t, dT_min, dT_avg, dT_max, rk_f, prog_f, num_stages]
"""







