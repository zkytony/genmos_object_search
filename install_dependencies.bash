set -o xtrace
pip install 'numpy>=1.20.3'
pip install 'opencv-contrib-python>=4.5.1.48'
pip install 'opencv-python>=4.1.1.26'
pip install 'pandas>=1.1.2'
pip install 'Pillow>=8.2.0'
pip install 'pygame>=1.9.6'
pip install 'scipy>=1.3.3'
pip install 'seaborn>=0.11.0'
# Note that we are trying out versions of the following packages
pip install pomdp-py
pip install nltk   # tested 3.7
pip install spacy  # tested 3.3.0
pip install torch  # tested 0.11.0
pip install torchvision  # tested 0.12.0
pip install sciex  # tested 0.3
python -m spacy download en_core_web_md
set +o xtrace
