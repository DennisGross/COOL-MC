# Install graphviy
sudo apt-get install graphviz -y
mkdir engine
cd engine
# Create Virtual-Envrionment
python -m venv env
source env/bin/activate
# Install PyCarl
git clone https://github.com/moves-rwth/pycarl.git
cd pycarl
python setup.py build_ext --jobs 1 develop
cd ..
# Install Stormpy
git clone https://github.com/moves-rwth/stormpy.git
cd stormpy
python setup.py build_ext --jobs 1 develop
# Install OpenAI Gym
pip install gym
pip install pyglet
# Install MLFlow
pip install mlflow
# Install PyTorch
pip install torch
# Install Pytest
pip install pytest
# Install Sklearn
pip install sklearn
# Install ipython
pip install ipython
# Install matplotlib
pip install matplotlib