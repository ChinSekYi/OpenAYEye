"""
reco_sys_custom_data.py

This module defines the RecoSysCustomData class, which handles the transformation and preprocessing 
of customer data for use in recommendation systems. The class maps categorical attributes to binary 
or encoded formats suitable for machine learning models and provides a method to retrieve the processed 
data as a Pandas DataFrame.

Classes:
    RecoSysCustomData: Maps customer information to machine-learning-ready data formats, including
                       handling binary and categorical encodings for fields like age, gender, income, 
                       region, country of residence, join channel, and others.

Methods:
    __init__(**kwargs): Initializes a RecoSysCustomData object with provided customer data.
    get_data_as_dataframe(): Returns the processed customer data as a Pandas DataFrame, encoded for model use.

Example:
    data = RecoSysCustomData(age=30, gender='Male', country_residence='Spain', ...)
    dataframe = data.get_data_as_dataframe()
"""


import pandas as pd
import sys
from src.exception import CustomException

class RecoSysCustomData:
    deceased_index_mapping = {'yes': 'S', 'no': 'N'}
    foreigner_index_mapping = {'yes': 'S', 'no': 'N'}
    residence_index_mapping = {'yes': 'S', 'no': 'N'}
    gender_mapping = {'Female': 'V', 'Male': 'H'}
    customer_relation_type_mapping = {'Individual': 'I', 'Associated': 'A'}
    country_residence_mapping = {
        'Argentina': 'AR', 'Austria': 'AT', 'Belgium': 'BE', 'Brazil': 'BR', 'Switzerland': 'CH',
        'Chile': 'CL', 'China': 'CN', 'Colombia': 'CO', 'Costa Rica': 'CR', 'Germany': 'DE',
        'Dominican Republic': 'DO', 'Ecuador': 'EC', 'Spain': 'ES', 'Finland': 'FI', 'France': 'FR',
        'United Kingdom': 'GB', 'Honduras': 'HN', 'India': 'IN', 'Italy': 'IT', 'Mexico': 'MX',
        'Mozambique': 'MZ', 'Nicaragua': 'NI', 'Netherlands': 'NL', 'Peru': 'PE', 'Poland': 'PL',
        'Portugal': 'PT', 'Sweden': 'SE', 'Taiwan': 'TW', 'United States': 'US', 'Venezuela': 'VE'
    }
    join_channel_mapping = {
        'Referral Program': 'KHE', 'Customer Service': 'KHD', 'Personal Banker': 'KFA',
        'Corporate Website': 'KHC', 'Mobile Advertising': 'KFC', 'Online Chat': 'RED',
        'Third-Party Website': 'KDH', 'Community Events': 'KHK', 'Online Campaigns': 'KEH',
        'Customer Rewards Program': 'KBG', 'Referral from Family/Friends': 'KHL',
        'Bank Promotions': 'KGC', 'In-Person Events': 'KHF', 'Loyalty Program': 'KFK',
        'Credit Union Partnership': 'KHN', 'University Partnership': 'KHA',
        'Trade Show': 'KHM', 'Sales Representative': 'KGX', 'Email Marketing': 'KFD',
        'Social Ads': 'KFG', 'Corporate Outreach': 'KCC', 'Government Program': 'KFJ',
        'Television Ads': 'KFL', 'Radio Ads': 'KFU', 'Newspaper Ads': 'KFS',
        'Magazine Ads': 'KAA', 'Neighborhood Event': 'KFP', 'Student Program': 'KFN',
        'Public Billboard': 'KGV', 'Community Board': 'KGY', 'Targeted Ads': 'KFF',
        'Digital Influencers': 'KDI', 'Local Partnerships': 'KLP', 'Affiliate Marketing': 'KAM',
        'Content Marketing': 'KCM', 'Online Forums': 'KOF', 'Corporate Sponsorships': 'KCS',
        'Pop-Up Events': 'KPE', 'Charity Events': 'KCE', 'Public Seminars': 'KPS',
        'Employee Advocacy': 'KEA', 'Co-Branding': 'KCB', 'Sponsored Content': 'KSC'
    }
    customer_segment_mapping = {'VIP': '01 - TOP', 'Private': '02 - PARTICULARES', 'University': '03 - UNIVERSITARIO'}
    region_mapping = {'East': 'EAST', 'North': 'NORTH', 'South': 'SOUTH', 'West': 'WEST'}
    new_customer_index_mapping = {'new customer': 1.0, 'existing customer': 0.0}
    customer_type_start_month = {
        'Jan': 1.0, 'Feb': 1.0, 'Mar': 1.0, 'Apr': 1.0, 'May': 1.0, 
        'Jun': 1.0, 'Jul': 1.0, 'Aug': 1.0, 'Sep': 1.0, 'Oct': 1.0, 
        'Nov': 1.0, 'Dec': 1.0
    }
    primary_customer_status_mapping = {
        'primary customer': 1.0,
        'non-primary customer': 99.0,
    }



    def __init__(self, **kwargs):
        self.age = kwargs.get("age")
        self.gender = self.gender_mapping.get(kwargs.get("gender"), kwargs.get("gender"))
        self.gross_income = kwargs.get("gross_income")
        self.customer_segment = self.join_channel_mapping.get(kwargs.get("customer_segment"), kwargs.get("customer_segment"))
        self.contract_length = kwargs.get("contract_length")
        self.seniority_months = kwargs.get("seniority_months")
        self.primary_customer_status = self.primary_customer_status_mapping.get(
            kwargs.get("primary_customer_status"), 
            kwargs.get("primary_customer_status") 
        )
        self.new_customer_index = kwargs.get("new_customer_index")
        month = kwargs.get("customer_type_start_month")
        self.customer_type_start_month = self.customer_type_start_month.get(month, "Jan")
        self.country_residence = self.country_residence_mapping.get(kwargs.get("country_residence"), kwargs.get("country_residence"))
        self.region = self.join_channel_mapping.get(kwargs.get("region"), kwargs.get("region"))
        self.join_channel = self.join_channel_mapping.get(kwargs.get("join_channel"), kwargs.get("join_channel"))
        self.deceased_index = self.deceased_index_mapping.get(kwargs.get("deceased_index"), kwargs.get("deceased_index"))
        self.foreigner_index = self.foreigner_index_mapping.get(kwargs.get("foreigner_index"), kwargs.get("foreigner_index"))
        self.residence_index = self.residence_index_mapping.get(kwargs.get("residence_index"), kwargs.get("residence_index"))
        self.customer_relation_type = self.customer_relation_type_mapping.get(kwargs.get("customer_relation_type"), kwargs.get("customer_relation_type"))
        new_customer_status = kwargs.get("new_customer_index")
        self.new_customer_index = self.new_customer_index_mapping.get(new_customer_status, new_customer_status)

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
            
            customer_segment_values = ['03 - UNIVERSITARIO', '02 - PARTICULARES', '01 - TOP']
           
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
