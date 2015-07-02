import pandas as pd
import numpy as np
import time
import os
import random

##################
#Module variables
##################

#Set identification and feature lists

#column names and order after transform_dfs_extracted_from_app_df()
trans_cols = ['job_id','stage_id','app_id','rk','n_stages','stage_dT']

features = ['dT_1','dT_f','dT_t','dT_1/dT_f',
			'dT_1/dT_t','dT_f/dT_t',
			'dT_min','dT_avg','dT_max',
			'rk_f','prog_f','n_stages']

features_w_ids = ['job_id','app_id']+features

#The schema required for any uploaded csv 
#Same as the internal sample schema
sample_cols = ['job_id', 'app_id', 'stage_id', 'rk', 'n_stages', 'stage_dT']

#Dictionary of "Contributing Factor" to churn likelihood messages depending on coefficient sign
#We only care about contributing feature for classes marked churn (i.e., prediction )
#If C_i > 0 then prediction([...,x_i,...]) increases as x_i -> infinity
#If C_i < 0 then prediction ([...,x_i,...]) increases as x_i -> 0
pos_coeff_msgs = {}
neg_coeff_msgs = {}

#positive coeff messages => LARGE feature value contributed to higher proedicted probability (i.e., churn)
pos_coeff_msgs['dT_1'] = 'Long first stage'
pos_coeff_msgs['dT_f'] = 'Long most recent stage'
pos_coeff_msgs['dT_t'] = 'Long total time'
pos_coeff_msgs['dT_1/dT_f'] = 'Long first stage time relative to most recent time'
pos_coeff_msgs['dT_1/dT_t'] = 'Long first stage time relative to total time'
pos_coeff_msgs['dT_f/dT_t'] = 'Long most recent stage relative to total time'
pos_coeff_msgs['dT_min'] = 'Long total time'	#reduce to total time
pos_coeff_msgs['dT_avg'] = 'Long average stage time'
pos_coeff_msgs['dT_max'] = 'Long total time'	#reduce to total time
pos_coeff_msgs['rk_f'] = 'Far along in pipeline'
pos_coeff_msgs['prog_f'] = 'Far along in pipeline'
pos_coeff_msgs['n_stages'] = 'Long pipeline'

#negative coeff messages => SMALL feature value contributed to higher predicted probability (i.e., churn)
neg_coeff_msgs['dT_1'] = 'Short first stage'
neg_coeff_msgs['dT_f'] = 'Short most recent stage'
neg_coeff_msgs['dT_t'] = 'Short total time'
neg_coeff_msgs['dT_1/dT_f'] = 'Short first stage time relative to most recent stage time'
neg_coeff_msgs['dT_1/dT_t'] = 'Short first stage time relative to total time'
neg_coeff_msgs['dT_f/dT_t'] = 'Short most recent stage relative to total time'
neg_coeff_msgs['dT_min'] = 'Short total time'		#reduce to total time
neg_coeff_msgs['dT_avg'] = 'Short average stage time'
neg_coeff_msgs['dT_max'] = 'Short total time'		#reduce to total time
neg_coeff_msgs['rk_f'] = 'Early in pipeline'
neg_coeff_msgs['prog_f'] = 'Early in pipeline'
neg_coeff_msgs['n_stages'] = 'Short pipeline'

TRAIN_PKG_PATH = os.path.dirname(__file__)
CSV_FOLDER = os.path.join(TRAIN_PKG_PATH, 'csv')

#TO DO: load DFs from SQL tables instead of CSV; create separate "CSV -> SQL" building script
def get_job_and_app_dfs_from_csvs(debug=False):
	"""
	Desc:
		Load job and app csv's into dataframes
		Correct timestamp values (parse_dates fails)

	Returns:
	Modified job and app dataframes
	"""
	app_df = pd.DataFrame.from_csv(os.path.join(CSV_FOLDER, 'app_data_20150609.csv'))
	app_df['entered_date'] = pd.to_datetime(app_df['entered_date'])
	app_df['exit_date'] = pd.to_datetime(app_df['exit_date'])

	#Rename jobs "dwell_time" so it doesn't conflict with app "stage" dwell_time during join
	job_df = pd.DataFrame.from_csv(os.path.join(CSV_FOLDER, 'job_data_20150609.csv'))
	job_df.rename(columns = {'dwell_time' : 'notify_time'}, inplace=True) 

	if debug:
		app_df = app_df.iloc[0:1000]

	return [job_df, app_df]

def extract_notify_and_churn_dfs_from_app_df(app_df):
	"""
	Desc: 
		Define bool arrays for extracting notified and churned samples
		Extract notified and churned samples which will have non-NaT stage_dT
	"""
	notify_bool = \
		((app_df['app_outcome'] == 'rejected') & \
		(app_df['rejection_type'] == 'we rejected them') & \
		(app_df['rejection_type_id'] == 1)) | \
		(app_df['app_outcome']=='hired')

	churn_bool = \
		(app_df['app_outcome'] == 'rejected') & \
		(app_df['rejection_type'] == 'they rejected us') & \
		(app_df['rejection_type_id'] == 2)

	#Don't keep any rows which will have NaT for when we define stage_dT
	entered_NaT_bool = app_df['entered_date'].isnull()
	exit_NaT_bool = app_df['exit_date'].isnull()
	stage_dT_NaT_bool = entered_NaT_bool | exit_NaT_bool

	#Extract df
	notify_df = app_df[notify_bool & (~stage_dT_NaT_bool)]
	churn_df = app_df[churn_bool & (~stage_dT_NaT_bool)]

	return [notify_df, churn_df]

def transform_dfs_extracted_from_app_df(dfs, job_df):
	"""
	Desc:
		Merge with jobs_df on job_id and app_id
		Define stage_dT
		Drop unnecessary columns coming from app and job dfs

	Params:
		@dfs: list of dfs to transform and clean

	Returns:
		list of modified dfs, in the same order
	"""
	new_dfs = []
	for df in dfs:
		#Join with job df
		df = pd.merge(df, job_df, on=['job_id','stage_id'])
		
		#Define stage_dT
		df['stage_dT'] = df['exit_date'] - df['entered_date']
		df['stage_dT'] = df['stage_dT'].apply(lambda x : x / np.timedelta64(1, 'D'))

		#Drop unecessary columns
		drop = ['entered_date','exit_date','app_outcome','notify_time','rejected_at','dwell_time','is_offer','rejection_type','rejection_type_id']
		df.drop(drop, axis=1, inplace=True)
		df.dropna(inplace=True)

		#Cutoff outlier values
		df = df[(df['rk'] > 0) & (df['rk'] < 10)]
		df = df[(df['n_stages'] > 0) & (df['n_stages'] < 10)]
		df = df[(df['stage_dT'] > 0) & (df['stage_dT'] < 30)]

		new_dfs.append(df)

	return new_dfs

def get_feature_dfs_from_transformed_dfs(dfs, rk_1_def=1):
	"""
	Desc:
		Feature vector represents an job_id and app_id grouping!

	Params:
		@dfs: list of dfs of transformed and cleaned class dataframes
        @rk_1_def: the rank value that will define the "first" stage, i.e., will determine dT_1

    Returns:
    	list of modified dfs with job_id, app_id
	"""
	new_dfs = []
	for df in dfs:

		#Rename columns to support other header names from uploaded csv's
		#Data must correspond to the order in trans_cols
		df.columns = trans_cols

		#Remove all stage entries with rk < rk_1_def
		df = df[df['rk'] >= rk_1_def]

		df_c = df.copy()
		df_c.index = [df_c['job_id'], df_c['stage_id'], df_c['app_id']] #multi-index

		gb = df.groupby(['job_id', 'app_id'])
		df = gb.agg({'stage_dT': ['sum', 'mean', 'min', 'max'], 'rk': {'rk_min': 'min', 'rk_max': 'max'}, 'n_stages': 'max'})

		#Select rank == rk_1_def and then group by job_id, app_id => should result in one sample per group
		#Take the first occuring one and call it dT_1 in case there are multiple by mistake
		dT_1 = df_c[df_c.rk == rk_1_def].groupby(['job_id','app_id']).stage_dT.first()
		df['dT_1'] = dT_1

		#Group df_c by job_id, app_id, rk => should have one sample per group
		#Take the first occuring one
		#dT_f has a multi-index over job_id, app_id, rk; rename the latter to rk_max for matching to df multi-index below
		dT_f = df_c.groupby(['job_id', 'app_id', 'rk']).stage_dT.first()
		dT_f.index.names = ['job_id', 'app_id', 'rk_max']

		#Create a column in df with (true!) max rk values using its multi-indexed column ['rk', 'rk_max']
		#Then, use this column to add an additional index in df's already multi-index over job_id, app_id
		df['rk_max'] = df['rk', 'rk_max']
		df.set_index('rk_max', append=True, inplace=True)

		#Now, df and dT_f have identically named multi-indices: job_id, app_id, rk_max
		#dT_f has more samples than df
			#df was gropued over only job_id, app_id (and then aggregated)
			#dT_f was grouped over job_id, app_id, and rk
		#But, we can thus create a dT_f column in df by setting it equal to dT_f
		#This works because it will only copy the entries into df where the multi-index agree's
		#df's last index rk_max contains true max rk values => will only pull the stage_dT values from dT_F corresponding to the max rank
		df['dT_f'] = dT_f 	

		#Renaming features
		#Name rk_max as rk_f
		#reset_index() and drop resulting rk_max column which was used as a multi-index marker (and is the same as rk_f)
		#drop rk_1 because it is the same for all samples by construction
		df.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in df.columns]
		col_dict = {'n_stages_max':'n_stages', \
					'stage_dT_sum':'dT_t', \
					'stage_dT_min':'dT_min', \
					'stage_dT_max':'dT_max', \
					'stage_dT_mean':'dT_avg', \
					'rk_rk_min':'rk_1', \
					'rk_rk_max':'rk_f'}
	    
		df.rename(columns=col_dict, inplace=True)
		df.reset_index(inplace=True)
		df.drop('rk_max', axis=1, inplace=True)
		df.drop('rk_1', axis=1, inplace=True)

		#Define ratio features
		df['prog_f'] = df['rk_f'] / df['n_stages']
		df['dT_1/dT_f'] = df['dT_1'] / df['dT_f']
		df['dT_1/dT_t'] = df['dT_1'] / df['dT_t']
		df['dT_f/dT_t'] = df['dT_f'] / df['dT_t']

		df = df[features_w_ids]
		df.dropna(inplace=True)

		new_dfs.append(df)

	return new_dfs

def add_outcomes_to_feature_dfs(feat_dfs, outcomes):
	"""
	Desc:
		Adds 'outcome' Series to feature dfs

	Params:
		@feat_dfs: list of individual class features dfs
		@outcomes: list of outcomes to assign to corresponding list of feature_dfs
	"""
	assert(len(feat_dfs) == len(outcomes))

	new_dfs = []
	for i in range(len(feat_dfs)):
		outcome_ser = pd.Series([outcomes[i]]*len(feat_dfs[i]), index=feat_dfs[i].index)
		feat_dfs[i]['outcome'] = outcome_ser
		new_dfs.append(feat_dfs[i])

	return new_dfs

def randomly_split_df_on_columns(df, cols, num_samples):
	"""
	Desc:
		Randomly splits a dataframe along unique combinations of vaules in given columns

	Params:
		@df: dataframe
		@cols: list of column labels along which to split the dataframe
		@num_id_samples: number of samples in the first sub dataframe
	"""

	all_ids = list(df.groupby(cols).grouper.groups.keys())	#list of tuples of unique combo's from given colums
	first_ids = random.sample(all_ids, num_samples)	
	second_ids = [ident for ident in all_ids if ident not in first_ids]
	first_dict = {}
	second_dict = {}

	#Create DataFrames from dictionaries with the given columns and values from the split lists
	for i in range(len(cols)):
	    first_dict[cols[i]] = [ident[i] for ident in first_ids]
	    second_dict[cols[i]] = [ident[i] for ident in second_ids]
	
	first_df = pd.DataFrame(first_dict)
	second_df = pd.DataFrame(second_dict)

	#Merge with the original df on the given columns to cut out samples corresponding to each group
	return df.merge(first_df), df.merge(second_df)

##################
#API methods
##################
def get_training_df(extract_sample=True, rk_1_def=1, debug=False):
	"""
	Desc:
		Main API method to build feature dataframes for each class that are ready for modeling
		All ETL, feature construction, outcome labeling, and concatination

		Supports extracting a sample before feature construction

	Params:
		@extract_sample: whether to extract samples from transformed df's before feature building
		@rk_1_def: the rank value that will define the "first" stage and dT_1, i.e., all lower stages will be dropped
		@debug: whether to use a very small fraction of app data for quick debugging

	Returns:
		List of two dataframes containing feature and outcome values for notify and churn groups, respectively
		DataFrame of transformed samples extract. Notify/churn balance of sample hard-coded here

	Notes:
		For training dataframes, features are constructed and outcomes are automatically added. Column is named 'outcome'
		Extracted sample dataframe is before feature construction to mimick upload schema support
		Number of notify/churn samples in extracted sample is hard-coded here during split
	"""

	#Build feature tables
	[job_df, app_df] = get_job_and_app_dfs_from_csvs(debug=debug)
	[notify_df, churn_df] = extract_notify_and_churn_dfs_from_app_df(app_df)
	[notify_df, churn_df] = transform_dfs_extracted_from_app_df([notify_df, churn_df], job_df)

	if extract_sample:
		notify_sample, notify_train = randomly_split_df_on_columns(notify_df, ['job_id','app_id'], 10)
		churn_sample, churn_train = randomly_split_df_on_columns(churn_df, ['job_id','app_id'], 10)

	else:
		#Otherwise use the whole data set for training
		notify_train = notify_df
		churn_train = churn_df

	[notify_feat, churn_feat] = get_feature_dfs_from_transformed_dfs([notify_train, churn_train], rk_1_def=rk_1_def)
	[notify_feat, churn_feat] = add_outcomes_to_feature_dfs([notify_feat, churn_feat], [0,1])
	train_df = pd.concat([notify_feat, churn_feat])
	sample_df = pd.concat([notify_sample, churn_sample])

	return train_df, sample_df
