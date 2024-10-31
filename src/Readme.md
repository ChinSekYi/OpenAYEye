# Python Pipeline Usages

## Recommendation System Model

### Overview
The Recommendation System model includes pipelines for data ingestion, training, and generating recommendations.

### Pipelines

#### 1. `Reco_sys_data_pipeline`
- **Purpose**: Ingest and transform raw data.
- **Functionality**:
  - Splits the combined dataset into training and test datasets.
- **Produces**: 2 datasets in the `artifacts` folder.
- **Output**:
    ```
    Data ingestion, cleaning and transformation completed for: data/recodataset.csv
    Training and test data saved at: ('artifacts/reco_sys_train_data.csv', 'artifacts/reco_sys_test_data.csv')
    ```

#### 2. `Reco_sys_model_trainer`
- **Purpose**: Train the model using `reco_sys_train_data.csv`.
- **Output**:
    ```
    OrderedDict([
        ('clicks', {'MSE': 2150661.38, 'R^2': 0.43, 'Adjusted R^2': 0.43}),
        ('leads', {'MSE': 706.71, 'R^2': 0.73, 'Adjusted R^2': 0.72}),
        ('orders', {'MSE': 9852.78, 'R^2': 0.81, 'Adjusted R^2': 0.81})
    ])
    ```

#### 3. `Reco_sys_predict_pipeline`
- **Purpose**: Generate ranking recommendations based on input values.
- **Input Example**:
    ```
    age  gender_H  gender_V  gross_income  customer_segment_03 - UNIVERSITARIO  ...  foreigner_index_N  residence_index_S  residence_index_N  customer_relation_type_I customer_relation_type_A
    0   92         1         0           500                                    0  ...                  1                  1                  0                         1                        0
    ```
- **Output**:
    ```
    Ranked Recommendations:
    1. Account
    2. Fixed Deposits
    3. Credit and Debit Card
    4. Loan
    ```

---

## ROI Model

### Overview
The ROI model consists of several Python pipelines designed for data ingestion, transformation, model training, and prediction.

### Pipelines

#### 1. `ROI_data_pipeline`
- **Purpose**: Ingest and transform raw data.
- **Functionality**:
  - Combines two cleaned datasets into one.
  - Splits the combined dataset into training and test datasets.
- **Produces**: 3 datasets in the `artifacts` folder.
- **Output**:
    ```
    Training and test data saved at: ('artifacts/roi_model_train_data.csv', 'artifacts/roi_model_test_data.csv')
    ```

#### 2. `ROI_model_trainer`
- **Purpose**: Train the model using `roi_clicks_train_data.csv`.
- **Output**:
    ```
    OrderedDict([
        ('clicks', {'MSE': 2150661.38, 'R^2': 0.43, 'Adjusted R^2': 0.43}),
        ('leads', {'MSE': 706.71, 'R^2': 0.73, 'Adjusted R^2': 0.72}),
        ('orders', {'MSE': 9852.78, 'R^2': 0.81, 'Adjusted R^2': 0.81})
    ])
    ```

#### 3. `ROI_clicks_data_pipeline_for_new_data.py`
- **Purpose**: Perform transformations on new data features (category and cost).
- **Output**: Training and test array, preprocessor object.

#### 4. `ROI_predict_pipeline`
- **Purpose**: Predict the number of clicks, leads, and orders based on input category and cost.
- **Input Example**:
    ```
    category   cost
    0   social  50000
    ```
- **Output**:
    ```
    Predicted Results:
    OrderedDict([('clicks', 4077.91), ('leads', 91.88), ('orders', 11.2)])
    ```

### Instructions to Run Pipeline
1. **Data Ingestion, Cleaning, and Transformation**:
    ```
    python3 -m src.components.roi.ROI_data_pipeline
    ```
2. **Model Training**:
    ```
    python3 -m src.components.roi.ROI_model_trainer
    ```
3. **Prediction on New Data**:
    ```
    python3 -m src.pipeline.roi.ROI_predict_pipeline
    ```

---

### Conclusion
This README provides an overview of the Python pipelines used in the Recommendation System and ROI models. Each section details the purpose and outputs of the pipelines, along with instructions for running them. Feel free to modify and expand upon these sections as necessary for your project!
