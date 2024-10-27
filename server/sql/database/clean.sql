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
FROM engagement;

USE transact;
SELECT san.customer_id, COUNT(san.credit_card)
FROM santender san
GROUP BY san.customer_id;
