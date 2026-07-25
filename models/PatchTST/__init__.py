"""
PatchTST — A Time Series is Worth 64 Words (ICLR 2023).

Usage:
    import sys
    sys.path.insert(0, 'models/PatchTST')

    from patch_tst_baseline import run_patchtst_baseline
    summary = run_patchtst_baseline(data_dir="GEFCom2014-L_V2/Load", task_id=15)

Or CLI:
    python models/PatchTST/patch_tst_baseline.py --task 15
"""

from PatchTST_backbone import PatchTST_backbone
from patch_tst_baseline import PatchTSTModel, run_patchtst_baseline
