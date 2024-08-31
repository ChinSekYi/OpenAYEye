SET NAMES utf8mb4;
SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';
SET @old_autocommit=@@autocommit;

USE mock;

--
-- Dumping data for table users
--

SET AUTOCOMMIT=0;
INSERT INTO users (id, name, email, age, phone, access) VALUES 
(1, "Jon Snow", "jonsnow@gmail.com", 35, "(665)121-5454", "admin"),
(2, "Cersei Lannister", "cerseilannister@gmail.com", 42, "(421)314-2288", "manager"),
(3, "Jaime Lannister", "jaimelannister@gmail.com", 45, "(422)982-6739", "user"),
(4, "Anya Stark", "anyastark@gmail.com", 16, "(921)425-6742", "admin"),
(5, "Daenerys Targaryen", "daenerystargaryen@gmail.com", 31, "(421)445-1189", "user"),
(6, "Ever Melisandre", "evermelisandre@gmail.com", 150, "(232)545-6483", "manager"),
(7, "Ferrara Clifford", "ferraraclifford@gmail.com", 44, "(543)124-0123", "user"),
(8, "Rossini Frances", "rossinifrances@gmail.com", 36, "(222)444-5555", "user"),
(9, "Harvey Roxie", "harveyroxie@gmail.com", 65, "(444)555-6239", "admin");

COMMIT;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
SET autocommit=@old_autocommit;
