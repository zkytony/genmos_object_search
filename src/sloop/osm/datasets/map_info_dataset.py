###############################################################
# This REPLACES the MapInfoDataset originally in SL_OSM_Dataset,
# with some changes.
################################################################
# In POMDP coordinate space, (x,y) the x axis is horizontal positive to the right.
# The y axis is vertical positive down.

import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from .SL_OSM_Dataset.mapinfo.constants import FILEPATHS

class MapInfoDataset:

    def __init__(self):
        # map from landmark symbol to grid cell coordinates that this
        # landmark occupies.
        self.landmarks = {}  # {map_name -> {symbol -> [(x,y)...]}}
        self._name_to_symbols = {}
        self._map_dims = {}  # {map_name -> (width, length)}
        self._pomdp_to_idx = {}
        self._idx_to_pomdp = {}
        self._symbol_to_maps = {} # {symbol -> [map_name]}
        self._symbol_to_synonyms = {}  # {map_name -> {symbol -> {name1, name2, ...}}}
        self._cardinal_to_limit = {} # {map_name -> {"N": (0.02200,0.400) ...}}
        # NOTE: cell here DOES NOT MEAN grid cell in cartesian coordinates.
        #       IT IS A LAT/LON CELL.
        self._idx_to_cell = {}  # idx -> {"nw": [41.82, -71.41]}, ...
        self._symbol_to_feats = {}
        self._excluded_symbols = {}  # map_name -> excluded symbols
        self._streets = {}  # map_name -> symbols that are streets
        self._cell_to_symbol = {}  # map_name -> {(x,y) -> symbol}
        self._symbol_to_name = {}  # map_name -> symbol -> landmark name

    def map_dims(self, map_name):
        return self._map_dims[map_name]

    def center_of_mass(self, landmark_symbol, map_name=None):
        footprint = self.landmark_footprint(landmark_symbol, map_name=map_name)
        mean = np.mean(np.array(footprint),axis=0)
        return tuple(np.round(mean).astype(int))

    def xyrange(self, landmark_symbol, map_name=None):
        footprint = np.array(self.landmark_footprint(landmark_symbol,
                                                     map_name=map_name))
        return (np.max(footprint[:, 0]) - np.min(footprint[:, 0])),\
            (np.max(footprint[:, 1]) - np.min(footprint[:, 1]))


    def landmark_footprint(self, landmark_symbol, map_name=None):
        if map_name is None:
            map_name = self.map_name_of(landmark_symbol)
            if type(map_name) == list:
                raise ValueError("Landmark symbol %s corresponds to multiple maps: %s"
                                 % (landmark_symbol, str(map_name)))
        return self.landmarks[map_name][landmark_symbol]


    def name_for(self, symbol, map_name=None):
        if map_name is None:
            map_name = self.map_name_of(symbol)
            if type(map_name) == list:
                raise ValueError("Landmark symbol %s corresponds to multiple maps: %s"
                                 % (symbol, str(map_name)))
        return self._symbol_to_name[map_name][symbol]

    def streets(self, map_name):
        return self._streets[map_name]

    def cell_to_landmark(self, map_name):
        """Returns a dictionary from cell -> symbol"""
        return self._cell_to_symbol[map_name]

    def landmark_at(self, map_name, cell):
        return self._cell_to_symbol[map_name].get(cell, None)

    # Use only one version of these index -> pomdp function because we
    # are merging the data together.
    def idx_to_pomdp(self, map_name, idx):
        ## WARNING> THIS MAKES NO SENSE TO ME BUT IT SEEEMS TO BE NECESSARY.
        ## SOMEHOW THE NYC POMDP COORDS ARE FLIPPED.
        if map_name in {"neighborhoods", "dorrance"}:
            return tuple(reversed(self._idx_to_pomdp[map_name][idx]))
        else:
            return self._idx_to_pomdp[map_name][idx]

    def pomdp_to_idx(self, map_name, pomdp_coord):
        ## WARNING> THIS MAKES NO SENSE TO ME BUT IT SEEEMS TO BE NECESSARY.
        ## SOMEHOW THE NYC POMDP COORDS ARE FLIPPED.
        # pomdp_coord = tuple(reversed(pomdp_coord))
        if map_name in {"neighborhoods", "dorrance"}:
            pomdp_coord = tuple(reversed(pomdp_coord))
            return self._pomdp_to_idx[map_name].get(pomdp_coord, None)
        else:
            return self._pomdp_to_idx[map_name].get(pomdp_coord, None)

    def map_loaded(self, map_name):
        return map_name in self._idx_to_pomdp

    def map_name_of(self, landmark_symbol):
        res = self._symbol_to_maps.get(landmark_symbol, None)
        if res is not None and len(res) == 1:
            return list(res)[0]
        else:
            return res

    @property
    def idx_to_cell(self):
        return self._idx_to_cell

    def cell_to_idx(self, latlon, map_name):
        # this is different
        # find the correct version in pixel2grid or use that!
        lat, lon = latlon
        for cell_idx, cell_coords in self._idx_to_cell[map_name].items():
            south, west, north, east = self.cell_limits_latlon(map_name, cell_idx)
            if west <= lon <= east and south <= lat <= north:
                idx = int(cell_idx)
                break
        return idx


    def cell_limits_latlon(self, map_name, idx):
        """Returns the South, West, North, East limits in lat/lon
        of the given OSM map grid cell at index `idx`.

        The returned value is in the order South, West, North, East"""
        cell_coords = self._idx_to_cell[map_name][idx]
        south = cell_coords["sw"][0]
        west = cell_coords["sw"][1]
        north = cell_coords["ne"][0]
        east = cell_coords["ne"][1]
        return south, west, north, east

    def axes_for(self, landmark_symbol):
        """Returns the major and minor axes of the landmark in lat/lon."""
        map_name = self.map_name_of(landmark_symbol)
        return self.symbol_to_feats[map_name][landmark_symbol][2],\
            self.symbol_to_feats[map_name][landmark_symbol][3]

    @property
    def name_to_symbols(self):
        return self._name_to_symbols

    @property
    def cardinal_to_limit(self):
        return self._cardinal_to_limit

    @property
    def symbol_to_feats(self):
        return self._symbol_to_feats

    def landmarks_for(self, map_name):
        return list(set(self.landmarks[map_name].keys()) - self._excluded_symbols[map_name])

    def load_by_name(self, map_name):
        """load a map from filepaths specified in FILEPATHS"""
        if map_name in FILEPATHS:
            if "idx_to_cell" in FILEPATHS[map_name]:
                return self.load_by_name_original_osm(map_name)
            else:
                # This will assume the map_name is registered with new organization;
                # The difference is:
                # - there is no *_idx_* files
                # - landmark is specified by grid coordinates directly TODO: change this to be metric.
                return self.load_by_name_new(map_name)
        else:
            raise ValueError(f"map {map_name} not found")

    ############################ new format loading ##################################
    def load_by_name_new(self, map_name):
        return self.load_new(map_name,
                             FILEPATHS[map_name]["name_to_symbols"],
                             FILEPATHS[map_name]["symbol_to_grids"],
                             FILEPATHS[map_name]["symbol_to_synonyms"],
                             FILEPATHS[map_name]["streets"],
                             FILEPATHS[map_name]["map_dims"],
                             FILEPATHS[map_name]["excluded_symbols"])

    def load_new(self, map_name,
                 name_to_symbols_fp,
                 symbol_to_grids_fp,
                 symbol_to_synonyms_fp,
                 streets_fp,
                 map_dims_fp,
                 excluded_symbols_fp):
        with open(map_dims_fp) as f:
            self._map_dims[map_name] = tuple(json.load(f))
        with open(name_to_symbols_fp) as f:
            name_to_symbols = json.load(f)
        with open(symbol_to_grids_fp) as f:
            symbol_to_grids = json.load(f)
        with open(symbol_to_synonyms_fp) as f:
            symbol_to_synonyms = json.load(f)
        with open(excluded_symbols_fp) as f:
            excluded_symbols = json.load(f)
        with open(streets_fp) as f:
            streets = json.load(f)

        self._idx_to_cell[map_name] = None
        self._cardinal_to_limit[map_name] = None
        self._pomdp_to_idx[map_name] = None
        self._idx_to_pomdp[map_name] = None
        self._symbol_to_feats[map_name] = {}
        self._symbol_to_name[map_name] = {}
        self._symbol_to_synonyms[map_name] = symbol_to_synonyms
        self._excluded_symbols[map_name] = set(excluded_symbols)
        self._streets[map_name] = set(streets)

        if map_name not in self.landmarks:
            self.landmarks[map_name] = {}
            self._cell_to_symbol[map_name] = {}

        for landmark_name in name_to_symbols:
            symbol = name_to_symbols[landmark_name]
            self._name_to_symbols[landmark_name] = symbol
            self._symbol_to_name[map_name][symbol] = landmark_name

        for symbol in symbol_to_grids:
            self.landmarks[map_name][symbol] = list(map(tuple, symbol_to_grids[symbol]))
            if symbol not in self._symbol_to_maps:
                self._symbol_to_maps[symbol] = set()
            self._symbol_to_maps[symbol].add(map_name)


    ############################ original osm loading ##################################
    def load_by_name_original_osm(self, map_name):
        """This is used by original OSM maps"""
        return self.load_original_osm(map_name,
                                      FILEPATHS[map_name]["name_to_idx"],
                                      FILEPATHS[map_name]["pomdp_to_idx"],
                                      FILEPATHS[map_name]["name_to_symbol"],
                                      FILEPATHS[map_name]["cardinal_to_limit"],
                                      FILEPATHS[map_name]["idx_to_cell"],
                                      FILEPATHS[map_name]["name_to_feats"],
                                      FILEPATHS[map_name]["excluded_symbols"],
                                      FILEPATHS[map_name]["streets"],
                                      FILEPATHS[map_name]["map_dims"])

    def load_original_osm(self,
                          map_name,
                          name_to_idx_fp,
                          pomdp_to_idx_fp,
                          name_to_symbol_fp,
                          limit_fp,
                          idx_to_cell_fp,
                          name_to_feats_fp,
                          excluded_symbols_fp,
                          streets_fp,
                          map_dims):
        """This is used by original OSM maps
        filepath (str) path to `name_to_idx_{location}.json` file.
        name_to_symbol (dict) maps from e.g. "Waterman Street" to "WatermanSt" symbol.
        """
        self._map_dims[map_name] = map_dims
        with open(name_to_symbol_fp) as f:
            name_to_symbols = json.load(f)
        with open(name_to_idx_fp) as f:
            name_to_idx = json.load(f)
        with open(pomdp_to_idx_fp) as f:
            pomdp_to_idx = json.load(f)
            idx_to_pomdp = {pomdp_to_idx[tup]:eval(tup)
                            for tup in pomdp_to_idx}
            pomdp_to_idx = {eval(tup):pomdp_to_idx[tup]
                            for tup in pomdp_to_idx}
        with open(limit_fp) as f:
            limit_dict = json.load(f)
        with open(idx_to_cell_fp) as f:
            idx_to_cell = json.load(f)
        with open(name_to_feats_fp) as f:
            name_to_feats = json.load(f)
        with open(excluded_symbols_fp) as f:
            excluded_symbols = json.load(f)
        with open(streets_fp) as f:
            streets = json.load(f)

        self._idx_to_cell[map_name] = idx_to_cell
        self._cardinal_to_limit[map_name] = limit_dict
        self._pomdp_to_idx[map_name] = pomdp_to_idx
        self._idx_to_pomdp[map_name] = idx_to_pomdp
        self._symbol_to_feats[map_name] = {}
        self._symbol_to_name[map_name] = {}
        self._excluded_symbols[map_name] = set(excluded_symbols)
        self._streets[map_name] = set(streets)

        if map_name not in self.landmarks:
            self.landmarks[map_name] = {}
            self._cell_to_symbol[map_name] = {}

        for landmark_name in name_to_idx:
            symbol = name_to_symbols[landmark_name]
            self.landmarks[map_name][symbol] = [self.idx_to_pomdp(map_name, idx)
                                                for idx in name_to_idx[landmark_name]]
            self._name_to_symbols[landmark_name] = symbol
            self._symbol_to_name[map_name][symbol] = landmark_name
            if symbol not in self._symbol_to_maps:
                self._symbol_to_maps[symbol] = set()
            self._symbol_to_maps[symbol].add(map_name)
            if landmark_name in name_to_feats:
                self._symbol_to_feats[map_name][symbol] = name_to_feats[landmark_name]

        for symbol in self.landmarks_for(map_name):
            for cell in self.landmark_footprint(symbol, map_name):
                self._cell_to_symbol[map_name][cell] = symbol
    #####################################################################

    def visualize(self, map_name, landmark_symbol, bg_path=None,
                  display=False, img=None, color=(128,128,205), res=25):
        if img is None:
            w, l = self._map_dims[map_name]
            img = MapInfoDataset.make_gridworld_image(w,
                                              l,
                                              res, bg_path=bg_path)
        footprint = self.landmark_footprint(landmark_symbol, map_name)
        img = MapInfoDataset.highlight_gridcells(img, footprint,
                                         res, color=color, alpha=0.8)
        center = mapinfo.center_of_mass(landmark_symbol, map_name)
        img = MapInfoDataset.highlight_gridcells(img, [center],
                                         res, color=tuple(c*0.7 for c in color), alpha=0.8)
        if display:
            self._display_img(img)
        return img

    def visualize_heatmap(self, city_png_path, landmark_symbol, heatmap, f_o_r, path=None):
        """
        Scaling code from:
        https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/layer_images.html
        """
        # make these smaller to increase the resolution
        dx, dy = 0.05, 0.05
        w, l = self._map_dims[self.map_name_of(landmark_symbol)]
        x = np.arange(0, l, dx)
        y = np.flip(np.arange(0, w, dy))
        X, Y = np.meshgrid(x, y)
        # print("X is: ", X)
        # print("Y is: ", Y)
        extent = np.min(x), np.max(x), np.max(y), np.min(y)
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(1, 1, 1)

        # 1) visualize the map
        map_png = mpimg.imread(city_png_path)
        im1 = plt.imshow(map_png, interpolation='nearest', extent=extent)

        # 2) visualize the involved landmark
        lm_locs = self.landmark_footprint(landmark_symbol)
        for x, y in lm_locs:
            curr_circle = plt.Circle(xy=(x + 0.5, y + 0.5), radius=0.5, color='r', fill=False)
            ax.add_artist(curr_circle)

        # TODO:and its center of mass
        # TODO: and its major, minor axes

        # 3) visualize the frame of reference
        x1, y1, ang = f_o_r
        # front vec
        u1, v1 = pol2cart(1.0, ang)
        # right vec
        u2, v2 = pol2cart(1.0, ang - (np.pi/2.0))
        plt.quiver([x1 + 0.5], [y1 + 0.5], [u1, u2], [v1, v2], color=['r','b'])

        # 4) visualize the heatmap
        im4 = plt.imshow(heatmap, alpha=.65, interpolation='kaiser', extent=extent) # interpolation='bilinear'

        # formatting:
        cbar = plt.colorbar(im4)
        major_ticks = np.arange(0, 21)
        minor_ticks = np.arange(0, 21)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.xaxis.tick_top()
        ax.set_yticks(np.flip(major_ticks))
        ax.set_yticks(np.flip(minor_ticks), minor=True)
        ax.grid(which='both')
        fig.tight_layout()
        # plt.show()
        plt.savefig(path)
        plt.clf()

    def _display_img(self, img):
        img = cv2.flip(img, 1)  # flip horizontally
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # rotate 90deg clockwise
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('map', img)
        cv2.waitKey()

    @classmethod
    def highlight_gridcells(cls, img, coords, res,
                            color=(255, 107, 107), alpha=None):
        img_paint = img
        if alpha is not None:
            img_paint = img.copy()
        radius = int(round(res / 2))
        for x,y in coords:
            cv2.circle(img_paint, (y*res + radius,
                                   x*res + radius), radius, color, thickness=-1)
        if alpha is not None:
            img = cv2.addWeighted(img_paint, alpha, img, 1 - alpha, 0)
        return img

    @classmethod
    def make_gridworld_image(cls, width, length, res,
                             state=None, bg_path=None, target_colors={}):
        """Returns an opencv image of the gridworld. If `state` is given,
        then the grid cells corresponding to the robot, obstacles, and
        target objects will be marked. If `bg_path` is provided, then
        a background image will be displayed."""
        # convenience
        w,l,r = width,length,res
        # Preparing 2d array
        arr2d = np.full((width, length), 0)  # free grids
        if state is not None:
            for objid in state.object_states:
                pose = state.object_states[objid]["pose"]
                if state.object_states[objid].objclass == "robot":
                    arr2d[pose[0], pose[1]] = 0  # free grid
                elif state.object_states[objid].objclass == "obstacle":
                    arr2d[pose[0], pose[1]] = 1  # obstacle
                elif state.object_states[objid].objclass == "target":
                    arr2d[pose[0], pose[1]] = objid  # target

        ## Preparing a background image
        if bg_path is not None:
            img = cv2.imread(bg_path, cv2.IMREAD_COLOR)
            img = cv2.flip(img, 1)  # flip horizontally
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # rotate 90deg clockwise
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (w*r, l*r))
        else:
            # Creating image
            img = np.full((w*r,l*r,3), 255, dtype=np.int32)

        ## draw the gridworld
        for x in range(w):
            for y in range(l):
                if arr2d[x,y] == 0:    # free
                    if bg_path is None:
                        # If we don't have a background, then plot
                        # a white square indicating free space
                        cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                      (255, 255, 255), -1)
                elif arr2d[x,y] == 1:  # obstacle
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  (40, 31, 3), -1)
                else:  # target
                    objid = arr2d[x,y]
                    if objid in target_colors:
                        color = target_colors[objid]
                    else:
                        color = (255, 165, 0)
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  color, -1)
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              (0, 0, 0), 1, 8)
        return img


# Utility functions
def cart2pol(x, y):
	rho = np.sqrt(x**2 + y**2)
	phi = np.arctan2(y, x)
	return(rho, phi)

def pol2cart(rho, phi):
	x = rho * np.cos(phi)
	y = rho * np.sin(phi)
	return(x, y)


if __name__ == "__main__":
    mapinfo = MapInfoDataset()

    mapinfo.load_by_name("austin")
    img = mapinfo.visualize("austin", "LavacaPlaza", bg_path=FILEPATHS["austin"]["map_png"], color=(205, 105, 100))
    img = mapinfo.visualize("austin", "ColoradoSt", bg_path=FILEPATHS["austin"]["map_png"], color=(205, 205, 100), img=img)
    img = mapinfo.visualize("austin", "TotalRestorationofTexas", bg_path=FILEPATHS["austin"]["map_png"], color=(128,128,205), img=img)
    img = mapinfo.visualize("austin", "MaikoSushi", bg_path=FILEPATHS["austin"]["map_png"], color=(100,100,100), img=img)
    mapinfo.visualize("austin", "NorwoodTower", bg_path=FILEPATHS["austin"]["map_png"], color=(105, 205, 100), img=img, display=True)
    cv2.destroyAllWindows()

    mapinfo.load_by_name("cleveland")
    img = mapinfo.visualize("cleveland", "West6thSt", bg_path=FILEPATHS["cleveland"]["map_png"], color=(205, 105, 100))
    img = mapinfo.visualize("cleveland", "West4thSt", bg_path=FILEPATHS["cleveland"]["map_png"], color=(205, 205, 100), img=img)
    img = mapinfo.visualize("cleveland", "WestSaintClairAvenue", bg_path=FILEPATHS["cleveland"]["map_png"], color=(205, 205, 150), img=img)
    img = mapinfo.visualize("cleveland", "JohnsonCourt", bg_path=FILEPATHS["cleveland"]["map_png"], color=(100, 100, 100), img=img)
    img = mapinfo.visualize("cleveland", "VelvetDog", bg_path=FILEPATHS["cleveland"]["map_png"], color=(200, 40, 120), img=img)
    mapinfo.visualize("cleveland", "MarionBuilding", bg_path=FILEPATHS["cleveland"]["map_png"], color=(105, 205, 100), img=img, display=True)
    cv2.destroyAllWindows()

    mapinfo.load_by_name("denver")
    img = mapinfo.visualize("denver", "PeabodyWhiteheadMansion", bg_path=FILEPATHS["denver"]["map_png"], color=(205, 105, 100))
    img = mapinfo.visualize("denver", "East11thAvenue", bg_path=FILEPATHS["denver"]["map_png"], color=(100, 105, 100), img=img)
    img = mapinfo.visualize("denver", "East12thAvenue", bg_path=FILEPATHS["denver"]["map_png"], color=(205, 205, 100), img=img)
    img = mapinfo.visualize("denver", "DennisSheedyMansion", bg_path=FILEPATHS["denver"]["map_png"], color=(100, 300, 100), img=img)
    img = mapinfo.visualize("denver", "TheRooseveltBuilding", bg_path=FILEPATHS["denver"]["map_png"], color=(105, 205, 100), img=img)
    img = mapinfo.visualize("denver", "LoganSt", bg_path=FILEPATHS["denver"]["map_png"], color=(175, 205, 175), img=img)
    mapinfo.visualize("denver", "GothamCityCondos", bg_path=FILEPATHS["denver"]["map_png"], color=(200, 40, 120), img=img, display=True)
    cv2.destroyAllWindows()

    mapinfo.load_by_name("honolulu")
    img = mapinfo.visualize("honolulu", "FortStreetMall", bg_path=FILEPATHS["honolulu"]["map_png"], color=(205, 105, 100))
    img = mapinfo.visualize("honolulu", "BethelSt", bg_path=FILEPATHS["honolulu"]["map_png"], color=(50, 105, 200), img=img)
    img = mapinfo.visualize("honolulu", "KamehamehaVPostOffice", bg_path=FILEPATHS["honolulu"]["map_png"], color=(50, 105, 200), img=img)
    img = mapinfo.visualize("honolulu", "BankofHawaiiHeadquartersMainBranch", bg_path=FILEPATHS["honolulu"]["map_png"], color=(50, 105, 200), img=img)
    img = mapinfo.visualize("honolulu", "MerchantSt", bg_path=FILEPATHS["honolulu"]["map_png"], color=(5, 200, 200), img=img)
    img = mapinfo.visualize("honolulu", "SouthKingSt", bg_path=FILEPATHS["honolulu"]["map_png"], color=(5, 5, 150), img=img)
    img = mapinfo.visualize("honolulu", "BankohParkingCenter", bg_path=FILEPATHS["honolulu"]["map_png"], color=(80, 180, 150), img=img)
    mapinfo.visualize("honolulu", "BishopSt", bg_path=FILEPATHS["honolulu"]["map_png"], color=(105, 205, 100), img=img, display=True)
    cv2.destroyAllWindows()

    mapinfo.load_by_name("washington_dc")
    img = mapinfo.visualize("washington_dc", "SupportBuildingServiceRoad", bg_path=FILEPATHS["washington_dc"]["map_png"], color=(205, 105, 100))
    img = mapinfo.visualize("washington_dc", "TheFStreetHouse", bg_path=FILEPATHS["washington_dc"]["map_png"], color=(100, 105, 100), img=img)
    img = mapinfo.visualize("washington_dc", "20thStreetNorthwest", bg_path=FILEPATHS["washington_dc"]["map_png"], color=(205, 205, 100), img=img)
    img = mapinfo.visualize("washington_dc", "FStreetNorthwest", bg_path=FILEPATHS["washington_dc"]["map_png"], color=(100, 300, 100), img=img)
    img = mapinfo.visualize("washington_dc", "Reiters/WashingtonLawandProfessionalBooks", bg_path=FILEPATHS["washington_dc"]["map_png"], color=(200, 40, 120), img=img)
    mapinfo.visualize("washington_dc", "HotelParking", bg_path=FILEPATHS["washington_dc"]["map_png"], color=(105, 205, 100), img=img, display=True)
    cv2.destroyAllWindows()

    mapinfo.load_by_name("nyc")
    img = mapinfo.visualize("nyc", "UniversityHall", bg_path=FILEPATHS["nyc"]["map_png"], color=(205, 105, 100))
    mapinfo.visualize("nyc", "Russell", bg_path=FILEPATHS["nyc"]["map_png"], color=(205, 205, 100), img=img, display=True)
    cv2.destroyAllWindows()

    mapinfo.load_by_name("faunce")
    img = mapinfo.visualize("faunce", "HopeCollege", bg_path=FILEPATHS["faunce"]["map_png"], color=(205, 105, 100))
    mapinfo.visualize("faunce", "SlaterHall", bg_path=FILEPATHS["faunce"]["map_png"], color=(12, 99, 100), img=img, display=True)
    cv2.destroyAllWindows()

    mapinfo.load_by_name("neighborhoods")
    img = mapinfo.visualize("neighborhoods", "HZ1A", bg_path=FILEPATHS["neighborhoods"]["map_png"], color=(205, 105, 100))
    img = mapinfo.visualize("neighborhoods", "HZ5F", bg_path=FILEPATHS["neighborhoods"]["map_png"], img=img, color=(4, 10, 123))
    img = mapinfo.visualize("neighborhoods", "HZ5C", bg_path=FILEPATHS["neighborhoods"]["map_png"], img=img, color=(4, 110, 123))
    img = mapinfo.visualize("neighborhoods", "HZ5B", bg_path=FILEPATHS["neighborhoods"]["map_png"], img=img, color=(4, 110, 123))
    img = mapinfo.visualize("neighborhoods", "HZ5D", bg_path=FILEPATHS["neighborhoods"]["map_png"], img=img, color=(114, 10, 223))
    img = mapinfo.visualize("neighborhoods", "PostSt", bg_path=FILEPATHS["neighborhoods"]["map_png"], img=img, color=(205, 125, 10))
    img = mapinfo.visualize("neighborhoods", "SunnyPl", bg_path=FILEPATHS["neighborhoods"]["map_png"], img=img, color=(23, 23, 250))
    img = mapinfo.visualize("neighborhoods", "MainSt", bg_path=FILEPATHS["neighborhoods"]["map_png"], img=img, color=(105, 20, 10))
    img = mapinfo.visualize("neighborhoods", "WestSt", bg_path=FILEPATHS["neighborhoods"]["map_png"], img=img, color=(54, 15, 140))
    img = mapinfo.visualize("neighborhoods", "EastSt", bg_path=FILEPATHS["neighborhoods"]["map_png"], img=img, color=(5, 125, 10))
    img = mapinfo.visualize("neighborhoods", "SouthSt", bg_path=FILEPATHS["neighborhoods"]["map_png"], img=img, color=(20, 125, 105))
    img = mapinfo.visualize("neighborhoods", "NorthSt", bg_path=FILEPATHS["neighborhoods"]["map_png"], img=img, color=(23, 100, 10))
    img = mapinfo.visualize("neighborhoods", "RiverDr", bg_path=FILEPATHS["neighborhoods"]["map_png"], img=img, color=(23, 200, 99))
    mapinfo.visualize("neighborhoods", "MainSt", bg_path=FILEPATHS["neighborhoods"]["map_png"], color=(30, 120, 100), img=img, display=True)
    cv2.destroyAllWindows()
