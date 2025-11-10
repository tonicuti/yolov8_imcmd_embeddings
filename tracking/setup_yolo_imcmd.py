import sys
from pathlib import Path
import ultralytics.nn.tasks as ytasks

def patch_yolo_custom_blocks(imcmd_dir: str):
    """
    Map các module custom (C2f_CA, AMFF, DynamicHeadLite) vào ultralytics.nn.tasks
    """
    imcmd_dir = Path(imcmd_dir)
    sys.path.insert(0, str(imcmd_dir.parent))
    from yolo_imcmd import modules as immods
    ytasks.C2f_CA = immods.C2f_CA
    ytasks.AMFF = immods.AMFF
    ytasks.DynamicHeadLite = immods.DynamicHeadLite
