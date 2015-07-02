from flask import render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import pickle
import os
import pandas as pd

from app import app
from app.training import etl			#contains feature list and contributing factor messages


##################
#PATHS
##################
APP_PKG_PATH = os.path.dirname(__file__)
ALLOWED_EXTENSIONS = ['csv','txt']
UPLOAD_FOLDER = os.path.join(APP_PKG_PATH, 'upload')

###################
#THRESHOLD
##################

THRESHOLD = 0.4

###################
#ERROR CODES
##################

ERROR_FAILED_UPLOAD = -1
ERROR_WRONG_SCHEMA = -2
ERROR_ALL_NANs = -3

def allowed_filename(filename):
	return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

def build_coeff_messages(coeffs):
	"""
	Select the appropriate 'contributing factor' message from etl.pos_coeff_msgs and etl.neg_coeff_msgs
	based on actual coefficient sign for the given model
	"""

	#All contributing feature messages for this particular model
	#by positive and negative coefficients from this model
	model_msgs = {}

	for i in range(len(coeffs)):
		if coeffs[i] > 0:
			model_msgs[etl.features[i]] = etl.pos_coeff_msgs[etl.features[i]]
		else:
			model_msgs[etl.features[i]] = etl.neg_coeff_msgs[etl.features[i]]

	return model_msgs

def set_std_scores(df, mins, ranges):
	df2 = pd.DataFrame(index = df.index, columns=df.columns)
	
	for i in range(len(etl.features)):
		df2[etl.features[i]] = (df[etl.features[i]]-mins[i])/ranges[i]
	
	return df2

@app.route('/upload', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		submitted_file = request.files['file']
		filename = secure_filename(submitted_file.filename)
		if submitted_file and allowed_filename(filename):
			file_path = os.path.join(UPLOAD_FOLDER, filename)
			submitted_file.save(file_path)

			#Redirect with code 302 to ensure method is GET otherwise can't return template
			return redirect(url_for('predict', internal_sample=False, filename=filename, uploaded=True), code=302)

		else:
			return redirect(url_for('predict', internal_sample=False, uploaded=False), code=302)

@app.route('/')
@app.route('/predict')
def predict():

	if request.args.get('internal_sample') == None:
		return render_template("predict.html", landing=True)

	pred_dicts = []

	if request.args.get('internal_sample') == None:
		print("fail")

	if request.args.get('uploaded') == 'False' and request.args.get('internal_sample') == 'False':
		error = ERROR_FAILED_UPLOAD

	#Load the dataframe from uploaded csv or internal sample pickle
	else:
		if request.args.get('uploaded') == 'True':
			filepath = os.path.join(UPLOAD_FOLDER, request.args.get('filename'))
			

		elif request.args.get('internal_sample') == 'True':
			filepath = os.path.join(APP_PKG_PATH, 'sample.csv')

		pred_df = pd.DataFrame.from_csv(filepath).dropna()
		pred_df.reset_index(inplace=True)

		if len(pred_df.columns) != len(etl.sample_cols):
			error = ERROR_WRONG_SCHEMA

		else:
			pred_feat = etl.get_feature_dfs_from_transformed_dfs([pred_df])[0] #this method takes and returns a list
			if len(pred_feat) == 0:
				error = ERROR_ALL_NANs

			else:
				error = 0

				model = pickle.load(open(os.path.join(APP_PKG_PATH, 'training/logreg.p'), 'rb'))
				stats = pickle.load(open(os.path.join(APP_PKG_PATH, 'training/stats.p'), 'rb'))

				pred_feat[etl.features] = set_std_scores(pred_feat[etl.features], stats[0], stats[1])
				
				model_preds = model.predict_proba(pred_feat[etl.features])
				churn_idx = model.classes_.tolist().index(1)
				coeffs_only = model.coef_.tolist()[0]

				coeffs = list(zip(etl.features, coeffs_only))
				print(coeffs)
				pos_coeffs = [coeff_tup for coeff_tup in coeffs if coeff_tup[1] > 0]
				neg_coeffs = [coeff_tup for coeff_tup in coeffs if coeff_tup[1] < 0]

				
				model_msgs = build_coeff_messages(coeffs_only)

				###################################
				#Significant Feature Contribution
				###################################
				for i in range(len(pred_feat)):
					current_pred = {}
					current_pred['job_id'] = int(pred_feat.iloc[i]['job_id'])
					current_pred['app_id'] = int(pred_feat.iloc[i]['app_id'])
					current_pred['prediction'] = model_preds[i][churn_idx]

					#Only attach messages to churned predictions 
					#because messages are built for "higher feature value leads to higher predicted probability"
					#i.e., churn
					if current_pred['prediction'] > THRESHOLD:
						curr_row = pred_feat.iloc[i]
						
						most_sig_pos_feat = pos_coeffs[0][0]
						most_sig_pos_coeff = pos_coeffs[0][1]
						max_pos_value = most_sig_pos_coeff*curr_row[most_sig_pos_feat]
						
						most_sig_neg_feat = neg_coeffs[0][0]
						most_sig_neg_coeff = neg_coeffs[0][1]
						max_neg_value = most_sig_neg_coeff*curr_row[most_sig_neg_feat]

						for coeff_tup in pos_coeffs:
							curr_feat = coeff_tup[0]
							curr_coeff = coeff_tup[1]
							curr_value = curr_coeff*curr_row[curr_feat]
							if curr_value < 0:
								print(curr_feat)
								print(curr_coeff)
								print(curr_value)
								print("NOOOOO!")

							if curr_value > max_pos_value:
								max_pos_value = curr_value
								most_sig_pos_feat = curr_feat
								most_sig_pos_coeff = curr_coeff

						for coeff_tup in neg_coeffs:
							curr_feat = coeff_tup[0]
							curr_coeff = coeff_tup[1]
							curr_value = curr_coeff*curr_row[curr_feat]
							if curr_value >= 0:
								print("NOOOOO!")

							if curr_value > max_neg_value:
								max_neg_value = curr_value
								most_sig_neg_feat = curr_feat
								most_sig_neg_coeff = curr_coeff

						if abs(most_sig_neg_coeff) > most_sig_pos_coeff:
							cont_feature = most_sig_neg_feat
						else:
							cont_feature = most_sig_pos_feat

						current_pred['cont_feat_msg'] = model_msgs[cont_feature]

					pred_dicts.append(current_pred)

				pred_dicts = sorted(pred_dicts, key = lambda k : k['prediction'], reverse=True)

	return render_template("predict.html", landing=False, error=error, pred_dicts=pred_dicts, threshold=THRESHOLD)

@app.route('/explore')
def explore():
	return render_template("explore.html")
