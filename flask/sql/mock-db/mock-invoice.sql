SET NAMES utf8mb4;
SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';
SET @old_autocommit=@@autocommit;

USE mock;
-- DROP TABLE IF EXISTS invoice;
--
-- Dumping data for table users
--

SET AUTOCOMMIT=0;
INSERT INTO invoice (invoice_id, name, email, cost,  phone, date) VALUES 
(1, "Jon Snow", "jonsnow@gmail.com", 21.24, "(665)121-5454", "2022-12-03"),
(2, "Jon Snow", "jonsnow@gmail.com", 43.24, "(665)121-5454", "2022-12-13"),
(3, "Cersei Lannister", "cerseilannister@gmail.com", 1.24, "(421)314-2288", "2021-06-15"),
(4, "Jaime Lannister", "jaimelannister@gmail.com", 11.24, "(422)982-6739", "2022-05-02"),
(5, "Anya Stark", "anyastark@gmail.com", 80.55, "(921)425-6742", "2022-03-21"),
(6, "Daenerys Targaryen", "daenerystargaryen@gmail.com", 1.24, "(421)445-1189", "2021-01-12"),
(7, "Ever Melisandre", "evermelisandre@gmail.com", 63.12, "(232)545-6483", "2022-11-02"),
(8, "Ferrara Clifford", "ferraraclifford@gmail.com", 52.42, "(543)124-0123", "2022-02-11"),
(9, "Rossini Frances", "rossinifrances@gmail.com", 21.24, "(222)444-5555", "2021-05-02"),
(10, "Cersei Lannister", "cerseilannister@gmail.com", 11.24, "(421)314-2288", "2021-07-15"),
(11, "Apple Banana", "apple_bana#gmail.com", 12.69, "(65)696-5555", "2024-08-31");

COMMIT;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
SET autocommit=@old_autocommit;
