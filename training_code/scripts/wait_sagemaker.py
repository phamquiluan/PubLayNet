import os
import sys
import json
import time
import datetime
import argparse
import logging

import boto3
import sagemaker


sm_session = sagemaker.Session() 
sm_session.logs_for_job(job_name=sys.argv[1], wait=True)
