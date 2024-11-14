-- set global sql_mode = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';
-- set global sql_mode=STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION
SET PERSIST sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''));
SET SESSION sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY','')); SELECT @@sql_mode;
DROP SCHEMA IF EXISTS transact;
CREATE SCHEMA transact; 

USE transact;

CREATE TABLE IF NOT EXISTS users (
	customer_id CHAR(10) PRIMARY KEY,
    person VARCHAR(32) NOT NULL, 
	current_age INTEGER,
	retirement_age INTEGER,
	-- birth_year_month TIMESTAMP,
	gender VARCHAR(32) NOT NULL,
	address VARCHAR(64) NOT NULL, 
	apartment INTEGER,
	city VARCHAR(32) NOT NULL,
	state VARCHAR(32) NOT NULL ,
	zipcode VARCHAR(32) NOT NULL, 
	latitude NUMERIC NOT NULL,
	longitude NUMERIC NOT NULL,
	per_capita_income NUMERIC NOT NULL,
	yearly_income NUMERIC NOT NULL,
	total_debt NUMERIC NOT NULL,
	fico_score NUMERIC NOT NULL,
	num_credit_cards INTEGER,
    deposits INTEGER,
    cards INTEGER, 
    account INTEGER,
    loan INTEGER
);


CREATE TABLE IF NOT EXISTS transactions (
    identifier INTEGER PRIMARY KEY,
    customer_id CHAR(10) NOT NULL REFERENCES user(customer_id),
    card_number VARCHAR(16) NOT NULL,
    date TIMESTAMP NOT NULL,
    amount NUMERIC NOT NULL,
    use_chip VARCHAR(32),
    merchant_name VARCHAR(32),
    merchant_city VARCHAR(32),
    merchant_state VARCHAR(32),
    zip VARCHAR(16),
    mcc INTEGER,
    errors MEDIUMTEXT,
    is_fraud VARCHAR(3)
);

CREATE TABLE IF NOT EXISTS churn (
    customer_id CHAR(10) PRIMARY KEY NOT NULL REFERENCES user(customer_id),
    churn_date TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS campaign (
    campaign_id VARCHAR(32) PRIMARY KEY NOT NULL,
    campaign_name VARCHAR(32),
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    target_segment VARCHAR(32) NOT NULL,
    budget NUMERIC NOT NULL,
    channel VARCHAR(32) NOT NULL,
    goal VARCHAR(32) NOT NULL,
    displays INTEGER
);

CREATE TABLE IF NOT EXISTS engagement (
    engagement_id CHAR(10) PRIMARY KEY,
    campaign_id VARCHAR(32) NOT NULL REFERENCES campaign(campaign_id),
    customer_id CHAR(10) NOT NULL REFERENCES user(customer_id),
    engagement_date TIMESTAMP NOT NULL,
    action_type VARCHAR(32) NOT NULL,
    device_type VARCHAR(32),
    feedback_score VARCHAR(32),
    conversion_value NUMERIC
);
