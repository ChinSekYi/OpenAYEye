DROP SCHEMA IF EXISTS transact;
CREATE SCHEMA transact;

USE transact;
SELECT *
FROM users;

USE transact;
SELECT *
FROM credit_cards;

USE transact;
SELECT *
FROM transactions;

USE transact;
SELECT *
FROM campaign;

USE transact;
SELECT *
FROM engagement;

USE transact;
SELECT san.customer_id, COUNT(san.credit_card)
FROM santender san
GROUP BY san.customer_id;

USE transact;
SELECT c.campaign_id,
c.campaign_name, 
c.start_date, c.end_date,
c.target_segment, 
c.budget AS mark_spent,
c.channel AS "category",
e.customer_id, 
e.engagement_date, 
e.action_type, e.device_type, e.feedback_score,
e.conversion_value
FROM campaign c, engagement e
WHERE c.campaign_id = e.campaign_id;

USE transact;
SELECT t1.campaign_id,
SUM(t1.budget) AS mark_spent,
t1.start_date,
t1.channel as category,
SUM(CASE WHEN t1.action_type = 'clicked' THEN 1 ELSE 0 END) AS clicks,
SUM(CASE WHEN t1.action_type = 'credentials' THEN 1 ELSE 0 END) AS leads, 
SUM(CASE WHEN t1.action_type = 'converted' THEN 1 ELSE 0 END) AS orders
FROM 
(SELECT c.campaign_id,
c.campaign_name, 
c.start_date, c.end_date,
c.target_segment, 
c.budget,
c.channel,
e.customer_id, 
e.engagement_date, 
e.action_type, e.device_type, e.feedback_score,
e.conversion_value
FROM campaign c, engagement e
WHERE c.campaign_id = e.campaign_id) AS t1
GROUP BY t1.campaign_id, t1.channel
ORDER BY t1.campaign_id, t1.start_date, t1.channel;
-- GROUP BY t1.campaign_id, t1.action_type;

USE transact;
SELECT c.campaign_id,
SUM(c.budget) AS mark_spent,
c.channel as category,
c.displays,
SUM(CASE WHEN e.action_type = 'clicked' THEN 1 ELSE 0 END) AS clicks,
SUM(CASE WHEN e.action_type = 'credentials' THEN 1 ELSE 0 END) AS leads, 
SUM(CASE WHEN e.action_type = 'converted' THEN 1 ELSE 0 END) AS orders
FROM campaign c, engagement e
WHERE c.campaign_id = e.campaign_id
GROUP BY c.campaign_id, c.channel, c.start_date;

USE transact;
SELECT c.campaign_id,
c.campaign_name, 
c.start_date, c.end_date,
c.target_segment, 
c.budget,
c.channel,
e.customer_id, 
e.engagement_date, 
e.action_type, e.device_type, e.feedback_score,
e.conversion_value
FROM campaign c, engagement e
WHERE c.campaign_id = e.campaign_id
LIMIT 100;

USE transact;
SELECT COUNT(DISTINCT e.customer_id)
FROM engagement e
WHERE e.action_type = 'converted'
AND e.engagement_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY);

USE transact;
SELECT COUNT(e.customer_id)
FROM engagement e
WHERE e.action_type IN ('credentials', 'clicked')
AND e.engagement_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY);

USE transact;
SELECT t1.converted / t2.impressions FROM 
(SELECT COUNT(DISTINCT e.customer_id) AS converted
FROM engagement e
WHERE e.action_type = 'converted'
AND e.engagement_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)) AS t1,
(SELECT COUNT(DISTINCT e.customer_id) AS impressions
FROM engagement e
WHERE e.action_type IN ('converted', 'credentials', 'clicked')
AND e.engagement_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)) AS t2;

USE transact;
SELECT COUNT(DISTINCT e.customer_id)
FROM engagement e
WHERE e.action_type IN ('converted', 'credentials', 'clicked')
AND e.engagement_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY);


USE transact;
SELECT  SUM(c.budget)
FROM campaign c
WHERE c.start_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
GROUP BY GROUP BY YEAR(c.start_date), MONTH(c.start_date)