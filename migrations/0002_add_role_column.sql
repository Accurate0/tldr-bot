CREATE TYPE role_t AS ENUM ('user', 'admin');

ALTER TABLE users
     ADD role role_t NOT NULL DEFAULT 'user'::role_t;
