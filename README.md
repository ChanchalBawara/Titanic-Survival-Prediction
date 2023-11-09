# Titanic Survival Prediction - Flask Version

This Flask application predicts the likelihood of survival for Titanic passengers based on input details. It is a Flask adaptation of the original Django project.

## Table of Contents
  - [Installation](#installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Customization](#customization)

## Installation

1. Clone the repository:

```bash
   git clone https://github.com/ChanchalBawara/Titanic-Survival-Prediction.git
```

2. Navigate to the project directory:

```bash
   cd Titanic-Survival-Prediction
```

3. Install the required dependencies:

```bash
   pip install -r requirements.txt
```

## Usage

Run the Flask application:

```bash
   python app.py
```

- Open your web browser and go to http://localhost:5000/ to access the home page.

- Fill in the passenger details on the form and click "Get Predictions" to see the survival prediction on the result page.

## Project Structure

- app.py: The main Flask application file.
- templates: Contains HTML templates for the home page (index.html) and result page (result.html).
- data: Contains the survival data
- model: Contains the trained machine learning model (titanic_survival_ml_model.sav) and scaler (scaler.sav).

## Customization

- Modify the machine learning model or scaler: If you want to use a different model or scaler, replace the existing files in the model folder.
- Customize the HTML templates: You can modify the templates in the templates folder to change the look and feel of the web pages.
