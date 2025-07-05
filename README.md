**Note this project was done as a project for a computer vision class and was run on our class server**
## YOLO Model:
Our most recent and updated YOLO v11 script is located in the yolo_attempt8_25_epochs_batchsize_2. This folder contains the results,
the best model, and the script for running the model. To run this model you must source our custom environment using the command,
"source /cs/cs153/customenvs/julia-stephanie/env/bin/activate". To train the model cd into the yolo_attempt8_25_epochs_batchsize_2
folder and run "python3 yolov11_model_testing.py". Run the command "script" beforehand if you wish to save the terminal output log
to a text file. 

## Faster R-CNN Model:
Our most recent and updated Faster R-CNN script is located in the faster_rcnn_attempt2_detectron2 folder. This folder contains the 
results, best model, a script for downloading the dataset in the correct format and a script for running the model. Before 
running this model you must source the detectron2 env that Tim very helpfully made for us using the command "source 
/proj/tcb/detectron2/bin/activate". To train the model run "python faster_rcnn_model_training.py". Again run the script command to 
log the output to a text file.  Also run "tensorboard --logdir <output path> --port 6006" before running the script. This will set
up TensorBoard which you can use to visualize your results in a separate window. If using a server, make sure to  forward the 
port on your local machine. 

## Misc: 
For both of the models mentioned above we have commented out the lines used to download our dataset to the server and removed the 
associated API keys since the data exists in each folder under the name "underwater_plastics_og_data-1". Uncomment this out if you wish to download the data.

Our inference testing notebooks are located the in the inference_testing folder. Both of these scripts were uploaded to and ran 
seperately in Google Colab and not designed to be run on the server. If you wish to run either of these notebooks, you will need 
to upload their corresponding datasets to Colab along with the notebooks and potentially change the current file paths. 
