
from common import err_if_none_arg
from vis import plot_bn_vs_other_gradient_magnitudes, plot_accuracy, plot_loss
from args import GRAPH_PARSER
from os.path import join,isdir,isfile,basename
import os 
import pandas as pd
import logging 
import matplotlib.pyplot as plt 
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth


if __name__ == "__main__":
    args = GRAPH_PARSER.parse_args()
   
    err_if_none_arg(args.experiment_name,"experiment_name")


    experiment_root = join(args.experiments,args.experiment_name)


  
   
  
    # Below code does the authentication
    # part of the code
    gauth = GoogleAuth()
    
    # Creates local webserver and auto
    # handles authentication.
    gauth.LocalWebserverAuth()       
    drive = GoogleDrive(gauth)
    
    # replace the value of this variable
    # with the absolute path of the directory
    path = experiment_root   
    
    # iterating thought all the files/folder
    # of the desired directory
    for x in os.listdir(path):
    
        f = drive.CreateFile({'parents': [{'id': '1O_gD_vQCS6qLTpzMHAPMTKiNEQYqp9Fs'}]'title': x})
        f.SetContentFile(os.path.join(path, x))
        f.Upload()
    
        # Due to a known bug in pydrive if we 
        # don't empty the variable used to
        # upload the files to Google Drive the
        # file stays open in memory and causes a
        # memory leak, therefore preventing its 
        # deletion
        f = None
    