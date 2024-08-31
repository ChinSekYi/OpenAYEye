USE mock;
-- SELECT * FROM users;

-- SELECT * FROM invoice;

SELECT invoice.invoice_id, users.name, users.phone, users.email, cost, date
FROM users
CROSS JOIN invoice
WHERE users.name = invoice.name
AND users.email = invoice.email
AND users.phone = invoice.phone
ORDER BY invoice.invoice_id;