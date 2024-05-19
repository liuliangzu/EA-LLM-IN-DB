wget http://homepages.cwi.nl/~boncz/job/imdb.tgz
tar -zxvf imdb.tgz
sudo su - postgres
psql
CREATE DATABASE imdbload;
\c imdbload
\i /var/lib/pgsql/benchmark//schematext.sql;
\copy aka_name from '/var/lib/pgsql/benchmark/aka_name.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy aka_title from '/var/lib/pgsql/benchmark/aka_title.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy cast_info from '/var/lib/pgsql/benchmark/cast_info.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy char_name from '/var/lib/pgsql/benchmark/char_name.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy comp_cast_type from '/var/lib/pgsql/benchmark/comp_cast_type.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy company_name from '/var/lib/pgsql/benchmark/company_name.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy company_type from '/var/lib/pgsql/benchmark/company_type.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy complete_cast from '/var/lib/pgsql/benchmark/complete_cast.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy info_type from '/var/lib/pgsql/benchmark/info_type.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy keyword from '/var/lib/pgsql/benchmark/keyword.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy kind_type from '/var/lib/pgsql/benchmark/kind_type.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy link_type from '/var/lib/pgsql/benchmark/link_type.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy movie_companies from '/var/lib/pgsql/benchmark/movie_companies.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy movie_info from '/var/lib/pgsql/benchmark/movie_info.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy movie_info_idx from '/var/lib/pgsql/benchmark/movie_info_idx.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy movie_keyword from '/var/lib/pgsql/benchmark/movie_keyword.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy movie_link from '/var/lib/pgsql/benchmark/movie_link.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy name from '/var/lib/pgsql/benchmark/name.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy person_info from '/var/lib/pgsql/benchmark/person_info.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy role_type from '/var/lib/pgsql/benchmark/role_type.csv' with delimiter as ',' csv quote '"' escape as '\';
\copy title from '/var/lib/pgsql/benchmark/title.csv' with delimiter as ',' csv quote '"' escape as '\';
