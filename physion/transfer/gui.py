import os, sys, pathlib
import numpy as np
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from assembling.saving import get_files_with_extension, list_dayfolder

# important folders are those with screen_frames > 2-5 images
