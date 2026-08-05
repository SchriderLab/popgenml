# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

import os
import shutil
import subprocess
from setuptools import Distribution

# 1. Define paths
base_dir = os.path.abspath(os.path.dirname(__file__))
discoal_dir = os.path.join(base_dir, 'simulators', 'discoal')
bin_dir = os.path.join(base_dir, 'bin')

# 2. Force compilation BEFORE setup() is invoked
print("--- Compiling discoal C binary ---")
try:
    # Use make clean to prevent stale builds
    subprocess.check_call(['make', 'clean'], cwd=discoal_dir)
    subprocess.check_call(['make'], cwd=discoal_dir)
    
    executable_path = os.path.join(discoal_dir, 'discoal')
    
    os.system('cd {}'.format(base_dir))
    os.system('cp {} "$CONDA_PREFIX/bin/"'.format(executable_path))
except subprocess.CalledProcessError as e:
    print(f"Compilation failed: {e}")
    raise

# 3. Define the distribution as containing compiled binaries
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True
setup(
    name='popgenml',
    version='0.0.1',
    packages=find_packages(),
    package_data={
        'popgenml': ['slim/*', 'scripts/*']
    },
    include_package_data=True,
    
)
