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
    'Online Banking': 'KAB', 'Bank Branch': 'KAD', 'ATM': 'KAE', 'Telephone Banking': 'KAF',
    'Mobile App': 'KAG', 'Mail': 'KAH', 'Brokerage': 'KAI', 'Financial Advisor': 'KAJ',
    'Employee Referral': 'KAP', 'Employer Partnership': 'KAQ', 'Digital Marketing': 'KAR',
    'Direct Mail': 'KAT', 'Social Media': 'KAZ', 'Event Booth': 'KBG', 'Retail Partner': 'KCC',
    'Community Events': 'KDH', 'TV Advertising': 'KEH', 'Bank Staff Referral': 'KFA',
    'Newspaper Advertising': 'KFC', 'Public Seminar': 'KFD', 'Billboard Advertising': 'KFF',
    'Financial Seminar': 'KFG', 'Partner Network': 'KFJ', 'Flyers': 'KFK', 'Word of Mouth': 'KFL',
    'Trade Shows': 'KFN', 'Corporate Sponsor': 'KFP', 'Job Fairs': 'KFS', 'University Fair': 'KFU',
    'Government Program': 'KGC', 'Youth Program': 'KGV', 'Charity Events': 'KGX',
    'School Partnership': 'KGY', 'Hospitality Partnership': 'KHA', 'Concert Sponsorship': 'KHC',
    'Exhibitions': 'KHD', 'Sports Sponsorship': 'KHE', 'Discount Promotions': 'KHF',
    'Membership Offers': 'KHK', 'Radio Advertising': 'KHL', 'Career Expo': 'KHM',
    'Volunteer Event': 'KHN', 'Referral Program': 'RED'
}
customer_segment_mapping = {'VIP': '01 - TOP', 'Private': '02 - PARTICULARES', 'University': '03 - UNIVERSITARIO'}
region_mapping = {'East': 'EAST', 'North': 'NORTH', 'South': 'SOUTH', 'West': 'WEST'}



#justification for join_channel_mapping

# KAB - Online Banking

# K: Represents "Kiosk" or "Key" for join channels in the banking context.
# A: Indicates "Access" related to online services.
# B: Represents "Banking."
# KAD - Bank Branch

# K: "Kiosk" or "Key" as a category for banking channels.
# A: "Access" for physical branches.
# D: Represents "Division" or "Direct."
# KAE - ATM

# K: "Kiosk" referring to self-service machines.
# A: "Automated" service for ATMs.
# E: Represents "Equipment."
# KAF - Telephone Banking

# K: "Kiosk" indicating a channel type.
# A: "Access" for banking via phone.
# F: Stands for "Functionality."
# KAG - Mobile App

# K: "Kiosk" or a "Key" reference for mobile channels.
# A: "Access" referring to applications.
# G: Stands for "Gateway."
# KAH - Mail

# K: "Kiosk" for traditional mail channels.
# A: "Access" related to correspondence.
# H: Represents "Handling."
# KAI - Brokerage

# K: "Kiosk" for investment channels.
# A: "Access" to brokerage services.
# I: Stands for "Investments."
# KAJ - Financial Advisor

# K: "Kiosk" indicating advisory channels.
# A: "Access" to financial services.
# J: Represents "Judgment" or "Advice."
# KAP - Employee Referral

# K: "Kiosk" as a reference for referral channels.
# A: "Access" for employees.
# P: Represents "Promotion."
# KAQ - Employer Partnership

# K: "Kiosk" for partnership channels.
# A: "Access" to employers.
# Q: Stands for "Quality" of partnerships.
# KAR - Digital Marketing

# K: "Kiosk" indicating marketing channels.
# A: "Access" to digital resources.
# R: Represents "Reach."
# KAT - Direct Mail

# K: "Kiosk" for direct communication.
# A: "Access" related to mail campaigns.
# T: Represents "Targeting."
# KAZ - Social Media

# K: "Kiosk" for social engagement.
# A: "Access" to social platforms.
# Z: Represents "Zone" of engagement.
# KBG - Event Booth

# K: "Kiosk" indicating physical presence.
# B: Stands for "Booth" participation.
# G: Represents "Gathering."
# KCC - Retail Partner

# K: "Kiosk" indicating retail partnerships.
# C: Represents "Channel" for retail.
# C: Indicates "Collaboration."
# KDH - Community Events

# K: "Kiosk" for community engagement.
# D: Represents "Direct" involvement.
# H: Stands for "Harmony" with the community.
# KEH - TV Advertising

# K: "Kiosk" for advertising channels.
# E: Represents "Electronic" media.
# H: Stands for "Highlights."
# KFA - Bank Staff Referral

# K: "Kiosk" for internal referrals.
# F: Represents "Facilitated" referrals.
# A: Indicates "Assistance."
# KFC - Newspaper Advertising

# K: "Kiosk" for traditional media.
# F: Stands for "Formal" advertising.
# C: Represents "Channel" of communication.
# KFD - Public Seminar

# K: "Kiosk" indicating public engagement.
# F: Represents "Facilitated" sessions.
# D: Stands for "Discussion."
# KFF - Billboard Advertising

# K: "Kiosk" for outdoor advertising.
# F: Stands for "Formal" advertising.
# F: Represents "Focus" on visibility.
# KFG - Financial Seminar

# K: "Kiosk" indicating finance-related events.
# F: Stands for "Facilitated" discussions.
# G: Represents "Guidance."
# KFJ - Partner Network

# K: "Kiosk" for networking channels.
# F: Stands for "Facilitated" connections.
# J: Represents "Joining" forces.
# KFK - Flyers

# K: "Kiosk" for promotional materials.
# F: Stands for "Facilitated" distribution.
# K: Indicates "Key" information.
# KFL - Word of Mouth

# K: "Kiosk" for informal channels.
# F: Stands for "Facilitated" recommendations.
# L: Represents "Links" to customers.
# KFN - Trade Shows

# K: "Kiosk" for industry events.
# F: Stands for "Facilitated" presentations.
# N: Represents "Networking."
# KFP - Corporate Sponsor

# K: "Kiosk" for corporate engagement.
# F: Stands for "Facilitated" sponsorship.
# P: Represents "Partnership."
# KFS - Job Fairs

# K: "Kiosk" indicating recruitment channels.
# F: Stands for "Facilitated" hiring.
# S: Represents "Selection."
# KFU - University Fair

# K: "Kiosk" for educational outreach.
# F: Stands for "Facilitated" connections.
# U: Represents "University."
# KGC - Government Program

# K: "Kiosk" for government interactions.
# G: Stands for "Government" initiatives.
# C: Represents "Collaboration."
# KGV - Youth Program

# K: "Kiosk" indicating youth-focused channels.
# G: Stands for "Guidance" for youth.
# V: Represents "Vision."
# KGX - Charity Events

# K: "Kiosk" for charitable activities.
# G: Stands for "Giving."
# X: Represents "X-factor" in community service.
# KGY - School Partnership

# K: "Kiosk" for educational partnerships.
# G: Stands for "Guidance" in education.
# Y: Represents "Youth."
# KHA - Hospitality Partnership

# K: "Kiosk" for hospitality channels.
# H: Stands for "Hospitality."
# A: Represents "Association."
# KHC - Concert Sponsorship

# K: "Kiosk" for entertainment channels.
# H: Stands for "Hospitality" in events.
# C: Represents "Concert."
# KHD - Exhibitions

# K: "Kiosk" for showcase events.
# H: Stands for "Highlight."
# D: Represents "Display."
# KHE - Sports Sponsorship

# K: "Kiosk" for sports-related channels.
# H: Stands for "Highlight" of events.
# E: Represents "Engagement."
# KHF - Discount Promotions

# K: "Kiosk" for promotional offers.
# H: Stands for "Highlight."
# F: Represents "Financial" benefits.
# KHK - Membership Offers

# K: "Kiosk" for membership channels.
# H: Stands for "Highlight" of offers.
# K: Represents "Key" memberships.
# KHM - Loyalty Programs

# K: "Kiosk" for loyalty initiatives.
# H: Stands for "Highlight" of loyalty.
# M: Represents "Membership."
# KHN - Discount Coupons

# K: "Kiosk" for coupon offers.
# H: Stands for "Highlight."
# N: Represents "Negotiation."
# KHO - Surveys

# K: "Kiosk" for feedback channels.
# H: Stands for "Highlight" of feedback.
# O: Represents "Outcomes."
# KHP - Market Research

# K: "Kiosk" for research initiatives.
# H: Stands for "Highlight."
# P: Represents "Publications."
# KHS - Focus Groups

# K: "Kiosk" for gathering insights.
# H: Stands for "Highlight."
# S: Represents "Sessions."
# KHT - Financial Literacy Programs

# K: "Kiosk" for educational initiatives.
# H: Stands for "Highlight."
# T: Represents "Training."
# KHU - Stakeholder Meetings

# K: "Kiosk" for stakeholder engagement.
# H: Stands for "Highlight."
# U: Represents "Understanding."
# KHV - Advisory Councils

# K: "Kiosk" for advisory roles.
# H: Stands for "Highlight."
# V: Represents "Value."
# KHW - Community Advisory Panels

# K: "Kiosk" for community input.
# H: Stands for "Highlight."
# W: Represents "Wisdom."
# KHX - Open Houses

# K: "Kiosk" for public engagement.
# H: Stands for "Highlight."
# X: Represents "Experience."
# KHY - Financial Workshops

# K: "Kiosk" for educational events.
# H: Stands for "Highlight."
# Y: Represents "Yearly."
