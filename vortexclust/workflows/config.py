from pathlib import Path
import os

# === Project Root === #
# ROOT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = Path.cwd()

# === Directory Paths === #
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
DOC_DIR = os.path.join(ROOT_DIR, "docs")

# Optional: subfolders
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw") # reserved for later use with netCDF files
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed") # reserved for later use with processed netCDF files
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
ANIMATIONS_DIR = os.path.join(OUTPUT_DIR, "animations")

# === Global Constants === #
DEFAULT_TIMEZONE = "UTC"
DEFAULT_LATITUDE_BAND = (50, 90)  # Degrees North, e.g. Arctic region
DEFAULT_PRESSURE_LEVELS = [10, 50, 100]  # hPa for stratosphere

# === Plotting Defaults === #
PLOT_STYLE = "whitegrid"
COLOR_PALETTE = "viridis"
DPI = 150
FIGSIZE = (10, 10)

# === Logging Setup === #
import logging

LOG_LEVEL = logging.INFO
LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(message)s"
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

logger = logging.getLogger("vortexclust")

# === Environment Flags === #
DEBUG_MODE = False
VERBOSE = True

# === Utility Functions === #
def set_debug_mode(enabled: bool = True):
    global DEBUG_MODE
    DEBUG_MODE = enabled
    logger.setLevel(logging.DEBUG if enabled else LOG_LEVEL)
    logger.debug("Debug mode enabled." if enabled else "Debug mode disabled.")

