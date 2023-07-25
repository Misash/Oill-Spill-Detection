# pasos de instalacion para habilitar el uso de gpu en linux

echo ; echo;
echo Install Miniconda

curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

echo ; echo;
echo Create a conda environment

if [ "$(basename "$0")" = "bash" ]; then
    source ~/.bashrc
else
    source ~/.zshrc
fi

conda create --name tf python=3.9

conda deactivate
conda activate tf

echo ; echo;
echo GPU setup
echo ; echo;

nvidia-smi

conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo ; echo;
echo Install TensorFlow
echo ; echo;

pip install --upgrade pip

pip install tensorflow==2.10.*
echo ; echo;
echo  Verify install

echo  CPU
echo ; echo;
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
echo ; echo;

echo GPU
echo ; echo;
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

echo ; echo;
echo ; echo;


echo Instalacion de librerias
pip 3 install opencv-python matplotlib



echo cada vez que se use: \"conda activate tf\"

echo cada vez y para desactivar: \"conda deactivate\"
