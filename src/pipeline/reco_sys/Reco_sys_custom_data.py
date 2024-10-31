# reco_sys_custom_data.py

import pandas as pd
import sys
from src.exception import CustomException

class RecoSysCustomData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get_data_as_dataframe(self):
        try:
            custom_data_input = {"age": [self.age]}

            # List of unique gender values
            gender_values = ['H', 'V']  

            for gender in gender_values:
                custom_data_input[f"gender_{gender}"] = [
                    1 if self.gender == gender else 0
                ]

            custom_data_input["gross_income"] = [self.gross_income]
            
            customer_segment_values = ['03 - UNIVERSITARIO', '02 - PARTICULARES', '01 - TOP']  # List of unique customer segments
           
            for segment in customer_segment_values:
                custom_data_input[f"customer_segment_{segment}"] = [
                    1 if self.customer_segment == segment else 0
                ]
            custom_data_input["contract_length"] = [self.contract_length]
            custom_data_input["seniority_months"] = [self.seniority_months]
            custom_data_input["primary_customer_status"] = [self.primary_customer_status]

            # List of unique new customer index values
            new_customer_index_values = [0.0, 1.0]  

            for index in new_customer_index_values:
                custom_data_input[f"new_customer_index_{index}"] = [
                    1 if self.new_customer_index == index else 0
                ]

            custom_data_input['customer_type_start_month'] = [self.customer_type_start_month]
            
            country_values = ['ES', 'AT', 'NL', 'FR', 'GB', 'CL', 'CH', 'DE', 'DO', 'BE', 
                              'AR', 'VE', 'US', 'MX', 'BR', 'IT', 'EC', 'PE', 'CO', 
                              'HN', 'FI', 'SE', 'AL', 'PT', 'MZ', 'CN', 'TW', 'PL', 
                              'IN', 'CR', 'NI', 'AL'] #Removed 'CA', 'IE' bc not in training set
            
            for country in country_values:
                custom_data_input[f"country_residence_{country}"] = [1 if self.country_residence == country else 0]

            regions = ["CENTRAL", "NORTH", "SOUTH", "EAST", "WEST"]

            for region in regions:
                custom_data_input[f"region_{region}"] = [
                    1 if self.region == region else 0
                ]
                
            join_channel_values = [
                'KHE', 'KHD', 'KFA', 'KHC', 'KAT', 'KFC', 'KAZ', 'RED', 
                'KDH', 'KHK', 'KEH', 'KAD', 'KBG', 'KHL', 'KGC', 'KHF', 
                'KFK', 'KHN', 'KHA', 'KHM', 'KAF', 'KGX', 'KFD', 'KAG', 
                'KFG', 'KAB', 'KCC', 'KAE', 'KAH', 'KAR', 'KFJ', 'KFL', 
                'KAI', 'KFU', 'KAQ', 'KFS', 'KAA', 'KFP', 'KAJ', 'KFN', 
                'KGV', 'KGY', 'KFF', 'KAP'
            ] #removed KGN, KHO
            
            for channel in join_channel_values:
                custom_data_input[f"join_channel_{channel}"] = [1 if self.join_channel == channel else 0]

            # List of unique deceased_index values
            deceased_index_values = ['N', 'S']  

            for index in deceased_index_values:
                custom_data_input[f"deceased_index_{index}"] = [
                    1 if self.deceased_index == index else 0
                ]

            # List of unique foreigner_index values
            foreigner_index_values = ['S', 'N']  

            for index in foreigner_index_values:
                custom_data_input[f"foreigner_index_{index}"] = [
                    1 if self.foreigner_index == index else 0
                ]

            # Handle residence index
            residence_index_values = ['S', 'N']  # List of unique residence index values
            for index in residence_index_values:
                custom_data_input[f"residence_index_{index}"] = [
                    1 if self.residence_index == index else 0
                ]

            customer_relation_type_values = ['I', 'A']  #removed P

            for relation_type in customer_relation_type_values:
                custom_data_input[f"customer_relation_type_{relation_type}"] = [
                    1 if self.customer_relation_type == relation_type else 0
                ]

            return pd.DataFrame(custom_data_input)
        except Exception as e:
            raise CustomException(e, sys)
