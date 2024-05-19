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
