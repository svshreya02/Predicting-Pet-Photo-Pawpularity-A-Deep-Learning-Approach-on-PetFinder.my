# Predicting-Pet-Photo-Pawpularity-A-Deep-Learning-Approach-on-PetFinder.my

## Overview
This project aims to predict the "Pawpularity" of pet photos using a deep learning model. The dataset is sourced from PetFinder.my, containing pet photos and their corresponding popularity scores. The project involves data analysis, model training using PyTorch, and deploying the model with a Streamlit interface.

## Repository Structure
- Data Visualisation and Analysis of dataset.ipynb: Jupyter Notebook for initial data exploration and visualization.
- pytorch-paw-model_train.ipynb: Jupyter Notebook for training the deep learning model using PyTorch.
app.py: Streamlit script for deploying the trained model.

## Installation
- Clone the repository:
```python
git clone https://github.com/svshreya02/Predicting-Pet-Photo-Pawpularity-A-Deep-Learning-Approach-on-PetFinder.my.git
```
cd Predicting-Pet-Photo-Pawpularity-A-Deep-Learning-Approach-on-PetFinder.my
Set up a virtual environment and install dependencies:

sh
Copy code
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
Usage
Data Analysis:
Open Data Visualisation and Analysis of dataset.ipynb in Jupyter Notebook to explore and visualize the dataset.

Model Training:
Open pytorch-paw-model_train.ipynb in Jupyter Notebook to train the deep learning model. The notebook includes data preprocessing, model architecture, training, and evaluation steps.

Model Deployment:
Run app.py using Streamlit to deploy the trained model with an interactive interface. Ensure you have the necessary dependencies installed and configured:

sh
Copy code
streamlit run app.py
Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -m 'Add new feature').
Push to the branch (git push origin feature-branch).
Open a pull request.
License
This project is licensed under the Apache-2.0 License. See the LICENSE file for details.

Contact
For any questions or issues, please contact svshreya02.