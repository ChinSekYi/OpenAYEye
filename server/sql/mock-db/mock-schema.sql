
SET NAMES utf8mb4;
SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

DROP SCHEMA IF EXISTS mock;
CREATE SCHEMA mock;
USE mock;

--
-- Table structure for table `users`
--

CREATE TABLE users (
  id SMALLINT UNSIGNED NOT NULL UNIQUE,
  name VARCHAR(45) NOT NULL,
  email VARCHAR(45) NOT NULL,
  age SMALLINT UNSIGNED NOT NULL,
  phone CHAR(20) NOT NULL,
  access VARCHAR(45) NOT NULL,
  address VARCHAR(45) NOT NULL, 
  city VARCHAR(45) NOT NULL,
  zipCode VARCHAR(10) NOT NULL, 
  registrarId INT UNSIGNED NOT NULL,
  PRIMARY KEY  (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE invoice (
    invoice_id SMALLINT UNSIGNED NOT NULL PRIMARY KEY,
    name VARCHAR(45) NOT NULL REFERENCES users(name), 
    email VARCHAR(45) NOT NULL REFERENCES users(email),
    cost FLOAT UNSIGNED NOT NULL,
    phone CHAR(20) NOT NULL REFERENCES users(phone),
    date TIMESTAMP NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;

