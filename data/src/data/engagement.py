import pandas as pd
import random
from datetime import datetime, timedelta
from .campaign import campaign

random.seed(10)
# Define number of entries to generate
num_entries = int(1e+04)

# Helper functions
def random_date(start, end):
    """Generate a random datetime between `start` and `end`."""
    return start + timedelta(days=random.randint(0, (end - start).days))

# Define possible values for each column
campaign_ids = [f"{str(i).zfill(4)}" for i in range(1, len(campaign))]  # 50 campaign IDs
customer_ids = [f"{str(i).zfill(4)}" for i in range(0, 2000)]  # 2000 customer IDs
action_types = ["scrolled", "clicked", "credentials", "converted"]
device_types = ["mobile", "laptop", "desktop"]
feedback_scores = list(range(1, 6))  # Feedback scores from 1 to 5

# Define date range
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 10, 31)

# Define seasonal parameters
def seasonal_engagement_date():
    """Generate random dates with seasonal distribution: higher frequency in Q4 and Q1."""
    month_weights = {
        1: 1.1, 2: 1.0, 3: 0.8, 4: 0.7, 5: 0.7, 6: 1.2,
        7: 1.3, 8: 1.0, 9: 0.7, 10: 0.8, 11: 1.2, 12: 1.3
    }
    # Select a month based on weighted distribution
    month = random.choices(range(1, 13), weights=month_weights.values(), k=1)[0]
    day = random.randint(1, 28)  # To simplify, avoid edge cases in different months
    year = random.choice([2023, 2024]) 
    # if month <= 10 else 2023
    return datetime(year, month, day)

# Seasonal conversion values - higher during holiday season and beginning of year
def seasonal_conversion_value(action):
    """Assign higher conversion values in peak season months."""
    if action == "converted":
        # Use higher range during Q4 and Q1 (months 1, 11, 12)
        peak_months = [1, 11, 12]
        current_month = random.choice(peak_months) if random.random() < 0.6 else random.randint(1, 12)
        return round(random.uniform(200, 1000) if current_month in peak_months else random.uniform(0, 800), 2)
    else:
        return 0.0

# Generate seasonal data
seasonal_data = {
    "engagement_id": [i for i in range(num_entries)],
    "campaign_id": [random.choice(campaign_ids) for _ in range(num_entries)],
    "customer_id": [random.choice(customer_ids) for _ in range(num_entries)],
    "engagement_date": [seasonal_engagement_date() for _ in range(num_entries)],
    "action_type": [random.choice(action_types) for _ in range(num_entries)],
    "device_type": [random.choice(device_types) for _ in range(num_entries)],
    # "conversion_value": [seasonal_conversion_value(action) for action in data["action_type"]],
    "feedback_score": [random.choice(feedback_scores) for _ in range(num_entries)]
}

# Create DataFrame with seasonality
engagement = pd.DataFrame(seasonal_data)

# Define probabilities for action types based on campaign attributes
def action_type_probability(segment, budget, channel, goal):
    base_probs = {"scrolled": 0.6, "clicked": 0.3, "credentials": 0.1, "converted": 0.01}
    
    # Adjust probabilities based on target segment
    if segment == "High-income":
        base_probs["converted"] += 0.1  # Increase conversion chance
    
    # Adjust based on budget
    if budget > 20000:
        base_probs["clicked"] += 0.1  # Higher budget -> more clicks
        base_probs["credentials"] += 0.1  # Higher budget -> more sign-ups
        base_probs["converted"] += 0.1
    
    # Adjust based on channel
    if channel in ["social", "search", "sms", "email"]:
        base_probs["scrolled"] += 0.1  # increases chance of being ignored
        base_probs["clicked"] += 0.1  # increases chance of being ignored
        # base_probs["converted"] += 0.1
    
    # Adjust based on goal
    if goal == "conversion":
        base_probs["converted"] += 0.1  # Conversion goal boosts conversion likelihood
    elif goal == "awareness":
        base_probs["scrolled"] += 0.1
    
    # Normalize probabilities to sum to 1
    total = sum(base_probs.values())
    for key in base_probs:
        base_probs[key] = base_probs[key] / total
    
    return base_probs

# Update action_type in Campaign_Engagement based on related Campaign data
updated_action_types = []
for _, eng in engagement.iterrows():
    camp = campaign[campaign["campaign_id"] == eng["campaign_id"]].iloc[0]
    probs = action_type_probability(
        camp["target_segment"],
        camp["budget"],
        camp["channel"],
        camp["goal"]
    )
    # Choose action_type based on calculated probabilities
    action_type = random.choices(
        list(probs.keys()), weights=list(probs.values()), k=1
    )[0]
    updated_action_types.append(action_type)

engagement["action_type"] = updated_action_types
engagement["conversion_value"]= [seasonal_conversion_value(action) for action in engagement["action_type"]]

# engagement = engagement.sort_values(['engagement_date', 'customer_id']).reset_index(drop=True)

# print(engagement[['action_type']].value_counts())
# print(engagement.describe())
# engagement.to_csv("engagement.csv", index=False)
