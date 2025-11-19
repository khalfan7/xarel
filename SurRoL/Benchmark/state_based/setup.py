"""
SurRoL Package Setup

Installation Instructions:
--------------------------

For Conda environment:
    conda create -n surrol python=3.12
    conda activate surrol
    
    # Install PyTorch with CUDA support (adjust version for your CUDA)
    pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    
    # Install SurRoL package
    cd /path/to/SurRoL/Benchmark/state_based
    pip install -e .
    
    # Install optional RL dependencies
    pip install -e .[rl]

For venv environment:
    python3.12 -m venv /path/to/surrol_env
    source /path/to/surrol_env/bin/activate
    
    # Install PyTorch with CUDA support
    pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    
    # Install SurRoL package
    cd /path/to/SurRoL/Benchmark/state_based
    pip install -e .
    
    # Install optional RL dependencies
    pip install -e .[rl]

Note: PyTorch is NOT included in install_requires to allow manual CUDA version selection
"""

from logging import root
import os
import sys
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess


def check_directories():
    root_files = os.listdir('.')
    assert 'setup.py' in root_files and 'ext' in root_files, 'Installation NOT in the root directory of SurRol.'
    
    submodules = os.listdir('./ext')
    assert 'bullet3' in submodules, 'Submodule `bullet3` not found.'
    assert 'pybullet_rendering' in submodules, 'Submodule `pybullet_rendering` not found.'
    
    submodule_bullet3_fns = os.listdir('./ext/bullet3')
    assert 'setup.py' in submodule_bullet3_fns, 'Submodule `bullet3` not cloned.'

    submodule_pybullet_rendering_fns = os.listdir('./ext/pybullet_rendering')
    assert 'setup.py' in submodule_pybullet_rendering_fns, 'Submodule `pybullet_rendering` not cloned.'

def install_submodules():
    pass
    # # Check prerequisites of project directories
    # print('=== Check project directories')
    # check_directories()
    # print('  -- done')

    # # Install submodules
    # print('\n=== Install submodules')
    # os.chdir('./ext/bullet3')
    # bullet_root_dir = os.path.realpath('.')
    # subprocess.check_call([os.path.abspath(sys.executable), "setup.py", "install"])
    # print('  -- pybullet installed')
    # os.chdir('../pybullet_rendering')
    # subprocess.check_call([os.path.abspath(sys.executable), "setup.py", "install", "--bullet_dir", bullet_root_dir])
    # print('  -- pybullet_rendering installed')
    # os.chdir('../panda3d-kivy')
    # subprocess.check_call([os.path.abspath(sys.executable), "setup.py", "install"])
    # print('  -- panda3d-kivy installed')
    # os.chdir('../../')

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        install_submodules()

class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        install_submodules()


if __name__ == '__main__':
    setup(
        name='surrol',
        version='0.2.0',
        description='SurRoL: An Open-source Reinforcement Learning Centered and '
                    'dVRK Compatible Platform for Surgical Robot Learning',
        author='Med-AIR@CUHK',
        keywords='simulation, medical robotics, dVRK, reinforcement learning',
        packages=[
            'surrol',
        ],
        python_requires='>=3.7',
        install_requires=[
            # Core environment dependencies
            "gym==0.26.2",
            "numpy<2",  # CRITICAL: NumPy 2.x incompatible with PyBullet
            "scipy",
            "pandas",
            
            # Physics simulation
            "pybullet==3.2.7",
            
            # Image/video processing
            "imageio",
            "imageio-ffmpeg",
            "opencv-python",
            
            # Robot kinematics
            "roboticstoolbox-python",
            "sympy",
            
            # 3D rendering
            "panda3d==1.10.14",
            "trimesh",
            "kivymd",
            
            # Configuration management (for RL training)
            "hydra-core==1.3.2",
            "omegaconf==2.3.0",
            
            # Logging
            "colorlog",
            "termcolor",
            
            # Memory monitoring
            "psutil",
        ],
        cmdclass={
            'install': PostInstallCommand,
            'develop': PostDevelopCommand
        },
        extras_require={
            # Optional dependencies for advanced features
            "rl": [
                "mpi4py>=3.1.5",  # Parallel training
                "wandb>=0.9.1",   # Experiment tracking
            ],
            "dev": [
                "ipython",
                "jupyter",
            ]
        }
    )
    

