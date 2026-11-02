set -e
cd ..
pip install -r requirements.txt
cd ../x_base
pip install -e .
cd ../x_diffusers
pip install -e .
cd ../tools/thirdpart_model/seedvr2
bash seedvr2.sh
cd ../../setup
bash install_ffmpeg.sh