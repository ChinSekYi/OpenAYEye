USE mock;
-- SELECT * FROM users;

-- SELECT * FROM invoice;

SELECT *
FROM users
CROSS JOIN invoice
WHERE users.name = invoice.name
AND users.email = invoice.email
AND users.phone = invoice.phone;