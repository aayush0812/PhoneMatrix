# Implementation Guide: Smartphone Price Predictor

**Context for AI Assistant:** I need you to implement the following project from scratch based on these step-by-step instructions. Please generate the necessary code for each file mentioned below.

## Step 1: Project Setup
Create the standard directory structure for a Streamlit machine learning application.
1. Initialize a `requirements.txt` file with the following dependencies:
   - `streamlit`
   - `pandas`
   - `scikit-learn`
   - `joblib`
   - `numpy`

## Step 2: Data Preprocessing & Model Training (`train_model.py`)
Write a Python script to process the raw data, train the model, and export it. The dataset file is `device_specs_structured_dataset.csv`.
1. **Load Data:** Read the CSV using `pandas`.
2. **Text Extraction (Regex):** Create a function using the `re` module to extract the first continuous number from a string.
3. **Apply Extraction:** Apply this function to the following raw columns to create clean, numeric columns:
   - `ram_raw` -> `ram_gb`
   - `storage_raw` -> `storage_gb`
   - `battery_raw` -> `battery_mah`
   - `rear_camera_raw` -> `main_camera_mp`
4. **Target Variable:** Map `price_inr` to `price`.
5. **Clean Data:** Create a new DataFrame with only the cleaned numeric feature columns and the target column. Drop any rows with `NaN` values.
6. **Train/Test Split:** Split the data (80% training, 20% testing).
7. **Model Training:** Initialize a `RandomForestRegressor(n_estimators=100, random_state=42)` and fit it to the training data.
8. **Evaluation:** Print the Mean Absolute Error (MAE) on the test set.
9. **Export:** Use `joblib` to save the trained model as `smartphone_model.pkl`.

## Step 3: Streamlit User Interface (`app.py`)
Write the Streamlit application code to serve the model.
1. **Import Libraries:** `streamlit`, `pandas`, `numpy`, `joblib`.
2. **Load Model:** Load `smartphone_model.pkl`.
3. **Header:** Set a title like "📱 Smartphone Retail Price Predictor".
4. **Input Layout:** Use `st.columns(2)` to create a clean, two-column layout for the inputs.
5. **Input Fields:**
   - Column 1: `st.selectbox` for RAM (Options: 2, 4, 6, 8, 12, 16, 24) and Storage (Options: 32, 64, 128, 256, 512, 1024).
   - Column 2: `st.slider` for Battery Capacity (2000 to 7000) and Main Camera (8 to 200).
6. **Prediction Execution:** Create an `if st.button("Predict Price"):` block.
   - Inside the block, construct a Pandas DataFrame from the user inputs. **Ensure column names match exactly** what the model was trained on in Step 2.
   - Pass the DataFrame to `model.predict()`.
   - Display the result formatted nicely as INR currency using `st.success()`.

## Step 4: Execution Instructions
Provide me (the user) with the terminal commands to:
1. Run the `train_model.py` script to generate the `.pkl` file.
2. Run the `app.py` script locally using Streamlit.