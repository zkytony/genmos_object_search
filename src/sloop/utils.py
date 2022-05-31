from datetime import datetime as dt
import tarfile
import shutil
import os
import cv2
import random
import numpy as np
import subprocess

# Printing
def json_safe(obj):
    if isinstance(obj, bool):
        return str(obj).lower()
    elif isinstance(obj, (list, tuple)):
        return [json_safe(item) for item in obj]
    elif isinstance(obj, dict):
        return {json_safe(key):json_safe(value) for key, value in obj.items()}
    else:
        return str(obj)
    return obj

def smart_json_hook(obj):
    rv = {}
    for k, v in obj.items():
        if isinstance(v, str):
            try:
                rv[k] = eval(v)
            except ValueError:
                rv[k] = v
        elif isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i], str):
                    try:
                        v[i] = eval(v[i])
                    except ValueError:
                        pass
            rv[k] = v
        else:
            rv[k] = v
    return rv


#### File Utils ####
def save_images_and_compress(images, outdir, filename="images", img_type="png"):
    # First write the images as temporary files into the outdir
    cur_time = dt.now()
    cur_time_str = cur_time.strftime("%Y%m%d%H%M%S%f")[:-3]
    img_save_dir = os.path.join(outdir, "tmp_imgs_%s" % cur_time_str)
    os.makedirs(img_save_dir)

    for i, img in enumerate(images):
        img = img.astype(np.float32)
        img = cv2.flip(img, 0)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # rotate 90deg clockwise
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        save_path = os.path.join(img_save_dir, "tmp_img%d.%s" % (i, img_type))
        cv2.imwrite(save_path, img)

    # Then compress the image files in the outdir
    output_filepath = os.path.join(outdir, "%s.tar.gz" % filename)
    with tarfile.open(output_filepath, "w:gz") as tar:
        tar.add(img_save_dir, arcname=filename)

    # Then remove the temporary directory
    shutil.rmtree(img_save_dir)

##### Print utils #####
# Colors on terminal https://stackoverflow.com/a/287944/2893053
class bcolors:
    WHITE = '\033[97m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

    @staticmethod
    def disable():
        bcolors.WHITE   = ''
        bcolors.CYAN    = ''
        bcolors.MAGENTA = ''
        bcolors.BLUE    = ''
        bcolors.GREEN   = ''
        bcolors.YELLOW  = ''
        bcolors.RED     = ''
        bcolors.ENDC    = ''

    @staticmethod
    def s(color, content):
        """Returns a string with color when shown on terminal.
        `color` is a constant in `bcolors` class."""
        return color + content + bcolors.ENDC

def print_info(content):
    print(bcolors.s(bcolors.BLUE, content))

def print_note(content):
    print(bcolors.s(bcolors.YELLOW, content))

def print_error(content):
    print(bcolors.s(bcolors.RED, content))

def print_warning(content):
    print(bcolors.s(bcolors.YELLOW, content))

def print_success(content):
    print(bcolors.s(bcolors.GREEN, content))

def print_info_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.BLUE, content))

def print_note_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.GREEN, content))

def print_error_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.RED, content))

def print_warning_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.YELLOW, content))

def print_success_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.GREEN, content))
# For your convenience:
# from moos3d.util import print_info, print_error, print_warning, print_success, print_info_bold, print_error_bold, print_warning_bold, print_success_bold





###################### colors
def random_unique_color(colors, ctype=1):
    """
    ctype=1: completely random
    ctype=2: red random
    ctype=3: blue random
    ctype=4: green random
    ctype=5: yellow random
    """
    if ctype == 1:
        color = "#%06x" % random.randint(0x444444, 0x999999)
        while color in colors:
            color = "#%06x" % random.randint(0x444444, 0x999999)
    elif ctype == 2:
        color = "#%02x0000" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#%02x0000" % random.randint(0xAA, 0xFF)
    elif ctype == 4:  # green
        color = "#00%02x00" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#00%02x00" % random.randint(0xAA, 0xFF)
    elif ctype == 3:  # blue
        color = "#0000%02x" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#0000%02x" % random.randint(0xAA, 0xFF)
    elif ctype == 5:  # yellow
        h = random.randint(0xAA, 0xFF)
        color = "#%02x%02x00" % (h, h)
        while color in colors:
            h = random.randint(0xAA, 0xFF)
            color = "#%02x%02x00" % (h, h)
    else:
        raise ValueError("Unrecognized color type %s" % (str(ctype)))
    return color


def rgb_to_hex(rgb):
    r,g,b = rgb
    return '#%02x%02x%02x' % (int(r), int(g), int(b))

def hex_to_rgb(hx):
    """hx is a string, begins with #. ASSUME len(hx)=7."""
    if len(hx) != 7:
        raise ValueError("Hex must be #------")
    hx = hx[1:]  # omit the '#'
    r = int('0x'+hx[:2], 16)
    g = int('0x'+hx[2:4], 16)
    b = int('0x'+hx[4:6], 16)
    return (r,g,b)
