
import mysql.connector as sql
import pandas

db_connection = sql.connect(host='133.186.143.65', database='wspider', user='wspider', password='wspider00!q')
pandas.read_sql('SELECT * FROM table_name', con=db_connection)