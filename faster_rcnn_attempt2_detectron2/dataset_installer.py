from roboflow import Roboflow
rf = Roboflow(api_key="") #Removed API key
project = rf.workspace("plastic-detection-r16is").project("underwater_plastics_og_data")
version = project.version(1)
dataset = version.download("coco")
