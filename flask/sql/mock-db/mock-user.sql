SET NAMES utf8mb4;
SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';
SET @old_autocommit=@@autocommit;

USE mock;
-- DROP TABLE IF EXISTS users;
--
-- Dumping data for table users
--

SET AUTOCOMMIT=0;
INSERT INTO users (id, name, email, age, phone, access, address, city, zipCode, registrarId) VALUES 
(1, "Jon Snow", "jonsnow@gmail.com", 35, "(665)121-5454", "admin", "0912 Won Street, Alabama, SY 10001", "New York", "10001", 123512),
(2, "Cersei Lannister", "cerseilannister@gmail.com", 42, "(421)314-2288", "manager", "1234 Main Street, New York, NY 10001", "New York", "13151", 123512),
(3, "Jaime Lannister", "jaimelannister@gmail.com", 45, "(422)982-6739", "user", "3333 Want Blvd, Estanza, NAY 42125", "New York",  "87281", 4132513),
(4, "Anya Stark", "anyastark@gmail.com", 16, "(921)425-6742", "admin", "1514 Main Street, New York, NY 22298", "New York", "15551", 123512),
(5, "Daenerys Targaryen", "daenerystargaryen@gmail.com", 31, "(421)445-1189", "user", "11122 Welping Ave, Tenting, CD 21321", "Tenting", "14215", 123512),
(6, "Ever Melisandre", "evermelisandre@gmail.com", 150, "(232)545-6483", "manager", "1234 Canvile Street, Esvazark, NY 10001", "Esvazark", "10001", 123512),
(7, "Ferrara Clifford", "ferraraclifford@gmail.com", 44, "(543)124-0123", "user", "22215 Super Street, Everting, ZO 515234", "Evertin", "51523", 123512),
(8, "Rossini Frances", "rossinifrances@gmail.com", 36, "(222)444-5555", "user", "4123 Ever Blvd, Wentington, AD 142213", "Esteras", "44215", 512315),
(9, "Harvey Roxie", "harveyroxie@gmail.com", 65, "(444)555-6239", "admin", "51234 Avery Street, Cantory, ND 212412", "Colunza", "111234", 928397),
(10, "Enteri Redack", "enteriredack@gmail.com",  42, "(222)444-5555", "user", "4123 Easer Blvd, Wentington, AD 142213", "Esteras", "44215", 533215),
(11, "Steve Goodman", "stevegoodmane@gmail.com", 11, "(444)555-6239", "user",  "51234 Fiveton Street, CunFory, ND 212412", "Colunza", "1234", 92197);

COMMIT;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
SET autocommit=@old_autocommit;
