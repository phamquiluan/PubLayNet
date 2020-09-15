git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
pip install -U pip setuptools
python setup.py build_ext install
cd ../..
rm -rf cocoapi
