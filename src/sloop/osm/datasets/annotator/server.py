import threading
import webbrowser
import http.server
import math
from io import StringIO
from urllib.parse import urlsplit, parse_qs
from sloop.datasets.annotator.amt_csv_loader import *
from sloop.datasets.SL_OSM_Dataset.mapinfo.constants import FILEPATHS
from sloop.datasets.SL_OSM_Dataset.mapinfo.map_info_dataset import MapInfoDataset
from sloop.datasets.osm_scripts.map_utils import pixel2grid, grid2pixels

AMT_LOADER = AMTCSVLoader(spacy_model_name="en_core_web_md")

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    """The test example handler."""

    def do_upload_csv_action(self, params):
        line_start = int(params["line_start"][0])
        line_end = int(params["line_end"][0])
        keyword = None if "keyword" not in params else params["keyword"][0]

        length = int(self.headers.get_all('content-length')[0])
        data_string = self.rfile.read(length)
        data_string = data_string.decode("UTF-8")

        # Skip the webkit
        kept_lines = []
        for line in data_string.split("\n"):
            if line.startswith("---") or line.startswith("Content") or len(line) <= 1:
                print("Skipping: %s" % line)
                continue
            kept_lines.append(line)
        data_string = "\n".join(kept_lines)

        f = StringIO(data_string)
        reader = csv.DictReader(f, delimiter=',')
        samples_json = AMT_LOADER.get_result(reader, line_start, line_end, keyword=keyword)

        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.flush_headers()
        self.wfile.write(json.dumps(samples_json).encode())


    def do_save(self, params):
        length = int(self.headers.get_all('content-length')[0])
        data_string = self.rfile.read(length)
        data_string = data_string.decode("UTF-8")

        output = json.loads(data_string)

        img_width = output["metadata"]["img_width"]
        img_length = output["metadata"]["img_length"]
        pixel_origin = output["metadata"]["pixel_origin"]

        map_info = MapInfoDataset()

        # Go over each sample
        try:
            for sample in output["samples"]:
                # Create a field for frame of refs in pomdp coords
                map_name = sample["map_name"]
                if not map_info.map_loaded(map_name):
                    print("Loading map info for %s" % map_name)
                    map_info.load_by_name(map_name)

                sample["sg"]["frame_of_refs_grid"] = []
                for foref_pixel in sample["sg"]["frame_of_refs_pixels"]:
                    grid_coord = pixel2grid(foref_pixel[0], img_width,
                                            img_length, map_info, map_name, pixel_origin=pixel_origin)
                    sample["sg"]["frame_of_refs_grid"].append([grid_coord,
                                                               foref_pixel[1]])  # keep the angle (rad)
                    print("HAPPY!")
        except Exception as ex:
            print("500 Internal Server Error: %s" % str(ex))
            import traceback
            traceback.print_exc()
            self.send_response(500)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.flush_headers()
            self.wfile.write(json.dumps(output).encode())
            return

        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.flush_headers()
        self.wfile.write(json.dumps(output).encode())


    def do_POST(self):
        """Handle a post request by returning the square of the number."""
        split = urlsplit(self.path)
        params = parse_qs(split.query)
        action = split.path.split("/")[1]
        if action == "upload-csv":
            self.do_upload_csv_action(params)
        elif action == "save":
            self.do_save(params)

        # print(length)
        # data_string = self.rfile.read(length)
        # try:
        #     result = math.sqrt(float(data_string))
        # except:
        #     result = 'error'
        # self.send_response(200)
        # self.send_header("Content-type", "text/plain")
        # self.end_headers()
        # self.flush_headers()
        # self.wfile.write(str(result).encode())



FILE = 'webpage/index.html'
PORT = 8000
def open_browser():
    """Start a browser after waiting for half a second."""
    def _open_browser():
        webbrowser.open('http://localhost:%s/%s' % (PORT, FILE))
    thread = threading.Timer(0.5, _open_browser)
    thread.start()

def start_server():
    """Start the server."""
    server_address = ("localhost", PORT)
    server = http.server.HTTPServer(server_address, RequestHandler)
    server.serve_forever()

if __name__ == "__main__":
    open_browser()
    start_server()
