import os
from flask import Flask
from werkzeug import secure_filename


#############################
#Flask object initialization
#############################

#app flask object has to be created before importing views below
#because views calls "import app from app"
app = Flask(__name__)

########################
#Import package modules
########################

from app import views	#to set model variable for use in views.py
