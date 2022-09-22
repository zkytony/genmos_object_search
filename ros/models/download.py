import os
import gdown
import subprocess

os.makedirs("yolov5", exist_ok=True)
outfname = "yolov5_lab_custom.zip"
output = f"yolov5/{outfname}"
if not os.path.exists(output):
    print("Downloading YOLOv5 weights for lab custom objects")
    data_url = "https://drive.google.com/uc?id=1I2LVbNUcysxqEoZmNov6Yt9YBEHnHu96"
    gdown.download(data_url, output, quiet=False)
    cmd=f'''
cd yolov5
unzip {outfname}
rm {outfname}
cd ..
'''
    subprocess.check_output(cmd, shell=True)
else:
    print(f"{output} already exists")
