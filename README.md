# Mobile-Phone-Price-Prediction-Using-Machine-Learning
A machine learning model that can be trained on device data to accurately predict smartphone prices based on their features.

### Software

- Python 3.x
- pandas
- numpy
- scikit-learn

All required Python libraries can be installed using the `requirements.txt` file.

### Hardware

This project was developed and tested on:

- MacBook Pro with Apple M1 Pro chip
- macOS

The application should also run on other systems that support Python 3.

### Dataset

The dataset used in this project is included in the repository:

mobile_price_prediction_with_names.csv

The dataset contains smartphone specifications such as battery capacity, RAM, storage, camera resolution, screen size, processor cores, and 5G support.

The target variable is **price_USD**, which represents the smartphone price.

The Python script expects this dataset file to be located in the same directory as `mobile_price_model.py`.

## C2 - Instructions to Run the AI/ML Application

1. Clone the repository.

git clone [<repository_url>](https://github.com/kevinmarindev/Mobile-Phone-Price-Prediction-Using-Machine-Learning.git)

2. Navigate to the project directory.

cd d683-advanced-ai-and-ml

3. Create and activate a virtual environment.

python3 -m venv venv
source venv/bin/activate

4. Install the required dependencies.

pip install -r requirements.txt

5. Run the AI/ML model.

python mobile_price_model.py

The script will:

- load and preprocess the dataset
- train the machine learning model
- evaluate model performance
- apply cross-validation
- perform hyperparameter tuning

