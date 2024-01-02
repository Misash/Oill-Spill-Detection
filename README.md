
# Oil Spill Detection in the Ocean Using U-net Convolutional Neural Network Architecture

## Overview
This project presents an innovative approach to detecting oil in the ocean using Convolutional Neural Network (CNN) with U-net architecture. Developed in response to a significant oil spill in Ventanilla, Peru, our model employs satellite imagery for effective identification and analysis of marine oil spills.



## Key Features
- **Advanced Image Processing**: Utilizing PeruSat-1 satellite imagery, processed for optimal neural network training.
- **Customized CNN Models**: Tailored U-net models for accurate hydrocarbon segmentation.
- **Multiple Training Phases**: Includes initial training, reinforced learning with additional datasets, and comparative analysis of pre-trained models.

## Dependencies
- Python >= 3.8.10
- OpenCV
- NumPy
- Matplotlib
- TensorFlow

## Repository Structure
- `Architecture`: Contains the architectures of trained models.
- `DataSave`: Stores trained models and training history.
- `Dataset`: SAR satellite image dataset.
- `DatasetPeruSat1`: PeruSat-1 satellite image dataset.
- `main.ipynb`: Primary Jupyter Notebook for project demonstration.



## Getting Started
To run the `main.ipynb` notebook:

1. Ensure all dependencies are installed. You can install them using `pip install -r requirements.txt` if you have a requirements file.
2. Navigate to the folder containing `main.ipynb`.
3. Launch Jupyter Notebook or Jupyter Lab by running `jupyter notebook` or `jupyter lab` in your terminal or command prompt.
4. Open `main.ipynb` from the Jupyter interface.
5. Run the cells in sequence to observe the model training, evaluation, and visualization processes.

Note: The notebook is commented to guide you through each step of the process.

## Trained Models
- `SAR`: Initial model trained with SAR satellite images.
- `PER1`: Model trained with PeruSat1 satellite images.
- `SAR_PER1`: SAR model enhanced with PeruSat1 dataset.
- `PER1_SAR`: PER1 model improved using SAR image dataset.

## Performance and Evaluation
Models are rigorously evaluated using F1-Score metrics, ensuring high accuracy and reliability in hydrocarbon detection.


![image](https://github.com/Misash/Oill-Spill-Detection/assets/70419764/bf53623c-cd9e-4280-84fd-cb44d19b4f03)
![image](https://github.com/Misash/Oill-Spill-Detection/assets/70419764/2175ed1a-2f5b-4453-8285-cfe4fa52bfb3)