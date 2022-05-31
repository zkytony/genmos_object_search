import os
import gdown
import subprocess


outfname = "SL_OSM_Dataset.zip"
output = f"data/{outfname}"
if not os.path.exists(output):
    print("Downloading SL_OSM_Dataset")
    sl_osm_dataset_url = "https://drive.google.com/uc?id=1K1SRR3rHcM8Jndjhb-YTB5kqefDNYYbH"
    gdown.download(sl_osm_dataset_url, output, quiet=False)
    cmd=f'''
cd data
unzip {outfname}
rm {outfname}
cd ..
'''
    subprocess.check_output(cmd, shell=True)
else:
    print(f"{output} already exists")

outfname = "foref_models.zip"
output = f"models/{outfname}"
if not os.path.exists(output):
    print("Downloading FoR prediction models")
    foref_models_url = "https://drive.google.com/uc?id=1XfOUa0xtRstUxJHBdNmk4SLJw970-4vV"
    gdown.download(foref_models_url, output, quiet=False)
    cmd=f'''
cd models
unzip {outfname}
mv models/* .
rm -r models
rm {outfname}
cd ..
'''
    subprocess.check_output(cmd, shell=True)
else:
    print(f"{output} already exists")
