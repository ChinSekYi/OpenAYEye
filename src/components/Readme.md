## Python pipeline usages:

`ROI_data_pipeline`:
- Used to ingest and transform raw data
- Combines two cleaned datasets into one combined dataset
- Split combined dataset into train and test data
- Create 3 train-test datasets for clicks, leads and orders
- Produces: 8 datasets in artifacts folder
- Output: Train and test data file path

---
`ROI_clicks_model_trainer`:
- Model training on `roi_clicks_train_data.csv`
- Output: r2_score, mse_values  
    ``` 
    Output:
    {'r2_scores': [2150661.377], 'mse_values': [0.431]}
    ```


`ROI_clicks_data_pipeline_for_new_data.py`:
- Perform transformation on new data features (category and cost)
- Output: Training and test array, preprocessor object 

`ROI_clicks_predict_pipeline`:
- Enter category and cost
- Output: Numbber of predicted clicks   
    ```
    Input:
    category   cost
    0   social  50000

    Predicted Results (Number of clicks):
    ```


The same 3 files above is done for leads and orders too. 