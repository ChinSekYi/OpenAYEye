USE mock;
-- SELECT * FROM users;

-- SELECT * FROM invoice;

SELECT u.name, u.phone, u.email, i.invoice_id, i.cost, i.date
FROM users u, invoice i
WHERE u.name = i.name
AND u.email = i.email
AND u.phone = i.phone
ORDER BY i.invoice_id;