import mysql.connector
import pandas as pd
import psycopg2 as p2
from psycopg2 import sql
from sqlalchemy import create_engine

# Database connection details
config = {
    'user': 'guest',
    'password': 'ctu-relational',
    'host': 'relational.fel.cvut.cz',
    'port': '3306',
    'database': 'stats'
}

conn_totaltable = p2.connect(
        host="localhost",
        database="stats",
        user="pilotscope",
        password="pilotscope",
        port=5432
    )
cur_all = conn_totaltable.cursor()
# Connect to the database
try:
    conn = mysql.connector.connect(**config)
    print("Connection established")
except mysql.connector.Error as err:
    print(f"Error: {err}")
    exit(1)

# Create a cursor object
cursor = conn.cursor()

# Get list of tables
cursor.execute("SHOW TABLES")
tables = cursor.fetchall()
print(tables)
import pandas as pd
from sqlalchemy import create_engine

def insert_into_pg(tables, conn_params):
    # Create a connection to the PostgreSQL database using sqlalchemy
    engine = create_engine(f'postgresql+psycopg2://{conn_params["user"]}:{conn_params["password"]}@{conn_params["host"]}:{conn_params["port"]}/{conn_params["database"]}')
    
    with engine.connect() as connection:
        for table_name in tables:
            table_name = table_name[0]  # Extract table name from the tuple
            
            # Read table into a DataFrame
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, connection)
            
            # Write DataFrame to the same table in PostgreSQL
            # First, drop the table if it exists
            drop_query = f"DROP TABLE IF EXISTS {table_name}"
            connection.execute(drop_query)
            
            # Create the table structure based on the DataFrame
            create_query = pd.io.sql.get_schema(df, table_name)
            connection.execute(create_query)
            
            # Insert data into the table
            df.to_sql(table_name, connection, if_exists='append', index=False)
            
            print(f"Inserted data into {table_name} in PostgreSQL")

# Example usage
conn_params = {
    "user": "pilotscope",
    "password": "pilotscope",
    "host": "localhost",
    "port": "5432",
    "database": "stats"
}
  # List of tables to process
insert_into_pg(tables, conn_params)

def download_db(tables):
    # Export each table to CSV
    for table_name in tables:
        table_name = table_name[0]  # Extract table name from the tuple
        # Read table into a DataFrame
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, conn)
        # Save DataFrame to CSV
        csv_filename = f"{table_name}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Exported {table_name} to {csv_filename}")

def analyze_db(tables):
    results = []
    for table_name in tables:
        table_name = table_name[0]  # Extract table name from the tuple
        # Get list of columns
        cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        columns = cursor.fetchall()
        print(table_name,columns)
        for column in columns:
            #print(column)
            column_name = column[0]
            column_type = column[1]
            print(column)
            if column_name.count("Date"):
                print(column_name)
                    # Get min, max, and cardinality
                cursor.execute(f"SELECT MIN(UNIX_TIMESTAMP({column_name})), MAX(UNIX_TIMESTAMP({column_name})), COUNT({column_name}) FROM {table_name}")
                min_val, max_val, cardinality = cursor.fetchone()
                #print(min_val, max_val, cardinality)
                # Get number of unique values
                cursor.execute(f"SELECT COUNT(DISTINCT UNIX_TIMESTAMP({column_name})) FROM {table_name}")
                num_unique_values = cursor.fetchone()[0]

                # Get most common values and frequencies
                cursor.execute(f"SELECT UNIX_TIMESTAMP({column_name}), COUNT(*) AS freq FROM {table_name} GROUP BY {column_name} ORDER BY freq DESC LIMIT 5")
                most_common_values = cursor.fetchall()
                most_common_values_list = [val for val, freq in most_common_values if freq/cardinality > 0.1]
                most_common_freq_list = [round(freq/cardinality, 3) for val, freq in most_common_values if freq/cardinality > 0.1]
                if len(most_common_freq_list) == 0:
                    most_common_values_list = ""
                    most_common_freq_list = ""
                # Get histogram bounds (assuming numerical data for simplicity)
                segment_size = cardinality // 5
                histogram_bounds = []
                for i in range(5):
                    offset = i * segment_size
                    cursor.execute(f"SELECT UNIX_TIMESTAMP({column_name}) FROM {table_name} ORDER BY UNIX_TIMESTAMP({column_name}) LIMIT 1 OFFSET {offset}")
                    histogram_bounds.append(cursor.fetchone()[0])

                results.append({
                    'name':"st_{0}.{1}".format(table_name[0],column_name),
                    'min': min_val,
                    'max': max_val,
                    'cardinality': cardinality,
                    'num_unique_values': num_unique_values,
                    'most_common_values': most_common_values_list,
                    'most_common_freq': most_common_freq_list,
                    'histogram_bounds': histogram_bounds
                })
            else:
                if column_type.count('int') == 0 :
                    continue
                # Get min, max, and cardinality
                cursor.execute(f"SELECT MIN({column_name}), MAX({column_name}), COUNT({column_name}) FROM {table_name}")
                min_val, max_val, cardinality = cursor.fetchone()
                #print(min_val, max_val, cardinality)
                # Get number of unique values
                cursor.execute(f"SELECT COUNT(DISTINCT {column_name}) FROM {table_name}")
                num_unique_values = cursor.fetchone()[0]

                # Get most common values and frequencies
                cursor.execute(f"SELECT {column_name}, COUNT(*) AS freq FROM {table_name} GROUP BY {column_name} ORDER BY freq DESC LIMIT 5")
                most_common_values = cursor.fetchall()
                most_common_values_list = [val for val, freq in most_common_values if freq/cardinality > 0.1]
                most_common_freq_list = [round(freq/cardinality, 3) for val, freq in most_common_values if freq/cardinality > 0.1]
                if len(most_common_freq_list) == 0:
                    most_common_values_list = ""
                    most_common_freq_list = ""
                # Get histogram bounds (assuming numerical data for simplicity)
                segment_size = cardinality // 5
                histogram_bounds = []
                for i in range(5):
                    offset = i * segment_size
                    cursor.execute(f"SELECT {column_name} FROM {table_name} ORDER BY {column_name} LIMIT 1 OFFSET {offset}")
                    histogram_bounds.append(cursor.fetchone()[0])
            
                # Append results to the list
                results.append({
                    'name':"st_{0}.{1}".format(table_name[0],column_name),
                    'min': min_val,
                    'max': max_val,
                    'cardinality': cardinality,
                    'num_unique_values': num_unique_values,
                    'most_common_values': most_common_values_list,
                    'most_common_freq': most_common_freq_list,
                    'histogram_bounds': histogram_bounds
                })
    results_df = pd.DataFrame(results)
    #results_df.to_csv('database_statistics.csv', index=False)

#insert_into_pg(tables)
#analyze_db(tables)
# Close the cursor and connection
cur_all.close()
conn_totaltable.close()
cursor.close()
conn.close()
print("Connection closed")
