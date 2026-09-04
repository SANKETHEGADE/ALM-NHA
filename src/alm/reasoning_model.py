import os, sys
ml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ml-service', 'src'))
if ml_path not in sys.path:
    sys.path.insert(0, ml_path)

from alm.reasoning_model import *
