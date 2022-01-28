import sqlite3

# 1) we need a connection to our database

conn = sqlite3.connect('neurons.db')
# if database already doesn't exist, it will create it
# You can use a database in memory --> but not permanent
# conn = sqlite3.connect(':memory:')

# To create a table, first we need to create a cursor
c = conn.cursor()
# connect to our connectiion, and make a cursor instance

# when want to do things in database, have to execute some command

# Create a table
c.execute("""CREATE TABLE neurons (
    cell_name TEXT,
    mouse_name TEXT,
    avg_dff_trace REAL
)

""")

"""
5 SQLite data types

NULL
INTEGER <- as a whole number
REAL <- decimal
TEXT
BLOB <- an image
"""

# Commit our command
conn.commit()

# Close our connection
conn.close()
