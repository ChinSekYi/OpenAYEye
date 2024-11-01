
## Recommendation system
### sample user input:
```
user_input = {
        'age': 42,
        'gender': 'Female',
        'gross_income': 6600,
        'customer_segment': 'VIP',
        'contract_length': 31,
        'seniority_months': 12,
        'primary_customer_status': 'primary customer',
        'new_customer_index': 'new customer',
        'customer_type_start_month': "Jun",
        'country_residence': 'Chile',
        'region': 'North',
        'join_channel': 'Online Banking',
        'deceased_index': 'no',
        'foreigner_index': 'yes',
        'residence_index': 'no',
        'customer_relation_type': 'Individual',
        }
```

### Details for each attribute
```
age: int

gender: str ['Female', 'Male']

gross_income: float

customer_segment: str ['VIP', 'Private', 'University']

contract_length: int

seniority_months: int

primary_customer_status: str ['primary customer', 'non-primary customer']

new_customer_index: str [new customer, existing customer]

customer_type_start_month: str ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

country_residence: str ['Argentina', 'Austria', 'Belgium', 'Brazil', 'Switzerland',
    'Chile', 'China', 'Colombia', 'Costa Rica', 'Germany', 'Dominican Republic', 'Ecuador',
    'Spain', 'Finland', 'France', 'United Kingdom', 'Honduras', 'India', 'Italy', 'Mexico',
    'Mozambique', 'Nicaragua', 'Netherlands', 'Peru', 'Poland', 'Portugal', 'Sweden',
    'Taiwan', 'United States', 'Venezuela']

region: str ['North', 'South','East','West']

join_channel: str ['Referral Program', 'Customer Service', 'Personal Banker', 'Corporate Website',
    'Mobile Advertising', 'Online Chat', 'Third-Party Website', 'Community Events', 'Online Campaigns',
    'Customer Rewards Program', 'Referral from Family/Friends', 'Bank Promotions', 'In-Person Events',
    'Loyalty Program', 'Credit Union Partnership', 'University Partnership', 'Trade Show', 
    'Sales Representative', 'Email Marketing', 'Social Ads', 'Corporate Outreach', 'Government Program', 
    'Television Ads', 'Radio Ads', 'Newspaper Ads', 'Magazine Ads', 'Neighborhood Event', 'Student Program', 
    'Public Billboard', 'Community Board', 'Targeted Ads', 'Digital Influencers', 'Local Partnerships',
    'Affiliate Marketing', 'Content Marketing', 'Online Forums', 'Corporate Sponsorships', 'Pop-Up Events',
    'Charity Events', 'Public Seminars', 'Employee Advocacy', 'Co-Branding', 'Sponsored Content']

deceased_index: str ['yes', 'no']

foreigner_index: str ['yes', 'no']

residence_index: str ['yes', 'no']

customer_relation_type: str ['Individual', 'Associated']
```



## ROI model
### sample user input:
```
user_input = {
    "category": "social",
    "cost": "50000"
}
```


### Details for each attribute
```
category: str ['social', 'search', 'influencer', 'media']
cost: float
```