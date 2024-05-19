sudo yum install git make gcc
git clone https://github.com/gregrahn/tpch-kit.git
cd tpch-kit/dbgen
make MACHINE=LINUX DATABASE=POSTGRESQL
export DSS_CONFIG=/.../tpch-kit/dbgen
export DSS_QUERY=$DSS_CONFIG/queries
export DSS_PATH=/path-to-dir-for-output-files
dbgen -h 
# generate data
qgen -v -c -d -s 1 > tpch-stream.sql
# generate query

su postgres
cp dss.ddl dss.sql
psql
\i /pathA/dss.sql
copy call_center from '/home/ywb/Data/tpcds-kit-master/tools/data/call_center.dat' with delimiter as '|' NULL '';
copy catalog_page from '/home/ywb/Data/tpcds-kit-master/tools/data/catalog_page.dat' with delimiter as '|' NULL '';
copy catalog_returns from '/home/ywb/Data/tpcds-kit-master/tools/data/catalog_returns.dat' with delimiter as '|' NULL '';
copy catalog_sales from '/home/ywb/Data/tpcds-kit-master/tools/data/catalog_sales.dat' with delimiter as '|' NULL '';
copy customer from '/home/ywb/Data/tpcds-kit-master/tools/data/customer.dat' with delimiter as '|' NULL '';
copy customer_address from '/home/ywb/Data/tpcds-kit-master/tools/data/customer_address.dat' with delimiter as '|' NULL '';
copy customer_demographics from '/home/ywb/Data/tpcds-kit-master/tools/data/customer_demographics.dat' with delimiter as '|' NULL '';
copy date_dim from '/home/ywb/Data/tpcds-kit-master/tools/data/date_dim.dat' with delimiter as '|' NULL '';
copy dbgen_version from '/home/ywb/Data/tpcds-kit-master/tools/data/dbgen_version.dat' with delimiter as '|' NULL '';
copy household_demographics from '/home/ywb/Data/tpcds-kit-master/tools/data/household_demographics.dat' with delimiter as '|' NULL '';
copy income_band from '/home/ywb/Data/tpcds-kit-master/tools/data/income_band.dat' with delimiter as '|' NULL '';
copy inventory from '/home/ywb/Data/tpcds-kit-master/tools/data/inventory.dat' with delimiter as '|' NULL '';
copy item from '/home/ywb/Data/tpcds-kit-master/tools/data/item.dat' with delimiter as '|' NULL '';
copy promotion from '/home/ywb/Data/tpcds-kit-master/tools/data/promotion.dat' with delimiter as '|' NULL '';
copy reason from '/home/ywb/Data/tpcds-kit-master/tools/data/reason.dat' with delimiter as '|' NULL '';
copy ship_mode from '/home/ywb/Data/tpcds-kit-master/tools/data/ship_mode.dat' with delimiter as '|' NULL '';
copy store from '/home/ywb/Data/tpcds-kit-master/tools/data/store.dat' with delimiter as '|' NULL '';
copy store_returns from '/home/ywb/Data/tpcds-kit-master/tools/data/store_returns.dat' with delimiter as '|' NULL '';
copy store_sales from '/home/ywb/Data/tpcds-kit-master/tools/data/store_sales.dat' with delimiter as '|' NULL '';
copy time_dim from '/home/ywb/Data/tpcds-kit-master/tools/data/time_dim.dat' with delimiter as '|' NULL '';
copy warehouse from '/home/ywb/Data/tpcds-kit-master/tools/data/warehouse.dat' with delimiter as '|' NULL '';
copy web_page from '/home/ywb/Data/tpcds-kit-master/tools/data/web_page.dat' with delimiter as '|' NULL '';
copy web_returns from '/home/ywb/Data/tpcds-kit-master/tools/data/web_returns.dat' with delimiter as '|' NULL '';
copy web_sales from '/home/ywb/Data/tpcds-kit-master/tools/data/web_sales.dat' with delimiter as '|' NULL '';
copy web_site from '/home/ywb/Data/tpcds-kit-master/tools/data/web_site.dat' with delimiter as '|' NULL '';
