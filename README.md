# Cholelithiasis Dashboard

This repository provides tools and models for analyzing and predicting health outcomes related to cholelithiasis (gallstone disease). It includes datasets, machine learning models, and Jupyter notebooks for data processing, model training, and evaluation.

## Features

- **Data Analysis**: Explore and preprocess cholelithiasis-related datasets.
- **Model Training**: Train various machine learning models to predict health statuses associated with cholelithiasis.
- **Evaluation**: Assess model performance using appropriate metrics.

## Repository Structure

- `.devcontainer/`: Configuration files for development container setup.
- `Dashboard/`: Contains the main dashboard application files.
- `colelithiasis_dataset.xlsx`: Dataset containing relevant patient data.
- `encoder.pkl`: Pre-trained encoder for categorical data.
- `gda.pkl`: Pre-trained Gaussian Discriminant Analysis model.
- `health_status_classification.ipynb`: Notebook for classifying health statuses.
- `installation.ipynb`: Notebook with installation instructions and environment setup.
- `logo.png`: Project logo image.
- `lr_model.pkl`: Pre-trained Logistic Regression model.
- `requirements.txt`: List of required Python packages.
- `rf_boosted.pkl`: Pre-trained Boosted Random Forest model.
- `rf_model.pkl`: Pre-trained Random Forest model.
- `scaler.pkl`: Pre-trained data scaler for feature normalization.
- `svm_model_linear.pkl`: Pre-trained Support Vector Machine (linear kernel) model.
- `svm_model_poly.pkl`: Pre-trained Support Vector Machine (polynomial kernel) model.
- `svm_model_rbf.pkl`: Pre-trained Support Vector Machine (RBF kernel) model.
- `test_data.xlsx`: Dataset for model testing and validation.
- `training.ipynb`: Notebook for training machine learning models.

## Installation

To set up the environment and install the required packages, follow the instructions in the `installation.ipynb` notebook.

## Usage

1. **Data Preparation**: Use `health_status_classification.ipynb` to load and preprocess the dataset.
2. **Model Training**: Execute `training.ipynb` to train the machine learning models.
3. **Evaluation**: Assess the models using appropriate metrics as demonstrated in the notebooks.
4. **Dashboard**: Navigate to the `Dashboard/` directory for instructions on launching the interactive dashboard.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request with your enhancements.

## License

This project is licensed under the [MIT License](LICENSE).
