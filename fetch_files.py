from roboflow import Roboflow
rf = Roboflow(api_key="FX0t6K12dztT7aeguf2z")
project = rf.workspace("garv-jhangiani").project("vehicle-detection-zzrdd")
dataset = project.version(4).download("yolov5")