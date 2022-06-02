import cv2
import pytest

from sloop.osm.datasets import MapInfoDataset, FILEPATHS

def test_dataset_mapinfo():
    mapinfo = MapInfoDataset()
    mapinfo.load_by_name("austin")
    img = mapinfo.visualize("austin", "LavacaPlaza", bg_path=FILEPATHS["austin"]["map_png"], color=(205, 105, 100))
    img = mapinfo.visualize("austin", "ColoradoSt", bg_path=FILEPATHS["austin"]["map_png"], color=(205, 205, 100), img=img)
    img = mapinfo.visualize("austin", "TotalRestorationofTexas", bg_path=FILEPATHS["austin"]["map_png"], color=(128,128,205), img=img)
    img = mapinfo.visualize("austin", "MaikoSushi", bg_path=FILEPATHS["austin"]["map_png"], color=(100,100,100), img=img)
    mapinfo.visualize("austin", "NorwoodTower", bg_path=FILEPATHS["austin"]["map_png"], color=(105, 205, 100), img=img, display=True)
    cv2.destroyAllWindows()
