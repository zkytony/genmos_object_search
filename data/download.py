import os
import gdown
import subprocess

outfname = "SL_OSM_Dataset.zip"
output = f"./{outfname}"
if not os.path.exists(output):
    print("Downloading SL_OSM_Dataset")
    sl_osm_dataset_url = "https://drive.google.com/uc?id=1K1SRR3rHcM8Jndjhb-YTB5kqefDNYYbH"
    gdown.download(sl_osm_dataset_url, output, quiet=False)
    cmd=f'''
unzip {outfname}
'''
    subprocess.check_output(cmd, shell=True)
else:
    print(f"{output} already exists")
