# Project Idea & Walkthrough: Smartphone Retail Price Predictor

## 1. Project Overview
This project is an end-to-end Machine Learning web application designed for an end-semester evaluation. The objective is to build a regression model that predicts the estimated retail price of a smartphone in INR (₹) based on its core hardware specifications. Furthermore, the project includes a live interactive user interface to accept user inputs and display the model's predictions in real-time.

## 2. Problem Statement
The consumer technology market is flooded with smartphones featuring vastly different specifications. The goal of this project is to apply machine learning regression techniques to understand the relationship between key hardware components (RAM, Internal Storage, Battery Capacity, and Camera Megapixels) and the final retail price of the device.

## 3. The Dataset
The project utilizes a raw dataset named `device_specs_structured_dataset.csv`. 
* **Data Challenge:** The dataset contains raw text strings for hardware specifications (e.g., `"12 GB / 16 GB"` for RAM, `"4400 mAh"` for Battery). 
* **Preprocessing Need:** A crucial part of this project involves writing data cleaning functions (using regular expressions) to extract the primary numeric values from these strings before feeding them into the machine learning pipeline.

## 4. Tech Stack & Methodology
* **Machine Learning Task:** Supervised Learning (Regression)
* **Algorithm:** Random Forest Regressor (Chosen for its robust performance with tabular data and resistance to overfitting without extensive hyperparameter tuning).
* **Data Processing:** Python, `pandas`, `re` (Regex), `numpy`
* **Model Training:** `scikit-learn`
* **Model Serialization:** `joblib`
* **User Interface:** Streamlit (A pure-Python web framework designed for ML engineers to deploy live interactive apps quickly).

## 5. User Interaction Flow
1. The user opens the web application.
2. The user adjusts sliders and dropdowns representing smartphone specifications:
   - RAM (GB)
   - Internal Storage (GB)
   - Battery Capacity (mAh)
   - Main Camera (Megapixels)
3. The user clicks the "Predict Price" button.
4. The backend loads the pre-trained `.pkl` model, formats the input data, and returns the predicted price instantly on the UI.