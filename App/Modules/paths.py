"""Project paths. Resolved relative to App/<this-file's parent>."""
import os

_HERE          = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT      = os.path.dirname(_HERE)
PROJECT_ROOT   = os.path.dirname(_APP_ROOT)
TEMP_DIR       = os.path.join(PROJECT_ROOT, "temp")
SAVESTATE_DIR  = os.path.join(PROJECT_ROOT, "savestate")
SAVESTATE_FILE = os.path.join(SAVESTATE_DIR, "session.json")