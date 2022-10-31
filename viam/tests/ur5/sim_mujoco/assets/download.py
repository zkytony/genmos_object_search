import os
import gdown
import subprocess

outfname = "mujoco_ur5e_robotiq-85.zip"
output = f"{outfname}"
if not os.path.exists(output):
    print("Downloading meshes")
    meshes_url = "https://drive.google.com/uc?id=1kZVvMnY_LCLqfXzd4jDNjorfTHJWhTYN"
    gdown.download(meshes_url, output, quiet=False)
    cmd=f'''
unzip {outfname}
rm {outfname}
'''
    subprocess.check_output(cmd, shell=True)
else:
    print(f"{output} already exists")
