import os
import sys

# Make "import validation" work when pytest is started from any folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))