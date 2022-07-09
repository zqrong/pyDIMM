import subprocess
import os

name = 'pyDIMM'

cwd=os.path.dirname(os.path.realpath(__file__))
subprocess.Popen("make", cwd=os.path.join(cwd, 'clibs'))

from .class_DIMM import DIMM