bash install_terastitcher.sh
sudo apt update
sudo apt install -y python3 python3-pip
pip3 install virtualenv
python3 -m virtualenv -p python3 ~/colm_pipeline_env
. ~/colm_pipeline_env/bin/activate
pip install -r requirements.txt
pip install --pre SimpleITK --find-links https://github.com/SimpleITK/SimpleITK/releases/tag/v2.0rc1
