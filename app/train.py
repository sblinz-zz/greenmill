from training import etl
from training import logreg
import pickle
import datetime as dt
import os
import time

APP_PKG_PATH = os.path.dirname(__file__)

#######################################
#Load data and build class feature DFs
#######################################

extract_sample = True

print("Building Training Dataset...")
start = dt.datetime.now()
train_df, sample_df = etl.get_training_df(extract_sample=extract_sample, rk_1_def=1, debug=True)
stop = dt.datetime.now()
print("Dataset built: " + str(stop-start))

#Save extracted sample dataframe to csv and pickle
if extract_sample:
	sample_df.to_csv(os.path.join(APP_PKG_PATH, 'sample.csv'), index=False)

#############
#Train Model
#############
print("Training Logistic Regression...")
start = dt.datetime.now()
logreg.train_and_save_logreg(train_df, etl.features)
stop = dt.datetime.now()
print("Logistic Regression trained: " + str(stop-start))