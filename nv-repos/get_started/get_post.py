from google.cloud.sql.connector import Connector, IPTypes
import pymysql
import os
import sqlalchemy

def connect_bdd() -> sqlalchemy.engine.base.Engine:
    """
    Initializes a connection pool for a Cloud SQL instance of MySQL.

    Uses the Cloud SQL Python Connector package.
    """
    # Note: Saving credentials in environment variables is convenient, but not
    # secure - consider a more secure solution such as
    # Cloud Secret Manager (https://cloud.google.com/secret-manager) to help
    # keep secrets safe.

    instance_connection_name = 'glossy-precinct-371813:europe-west9:nicolas-vallot-mysqlbd' 
    db_user = "nicolas-vallot"  # e.g. 'my-db-user'
    db_pass = "nicolasvallot"  # e.g. 'my-db-password'
    db_name = "nicolas-vallot-bd"  # e.g. 'my-database'

    ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC

    connector = Connector(ip_type)

    def getconn() -> pymysql.connections.Connection:
        conn: pymysql.connections.Connection = connector.connect(
            instance_connection_name,
            "pymysql",
            user=db_user,
            password=db_pass,
            db=db_name,
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
        # ...
    )
    return pool


def fun_post(request):

    connection = connect_bdd().connect()
    sql_request = 'INSERT INTO nv_tab (val) VALUES (%s)'
    data = (request.get_json().get('val'))
    connection.execute(sql_request, data)
    connection.close()

    return 'ok'

def get_text_entries():

    engine = connect_to_cloud_sql()
    connection = engine.connect()

    sql = 'SELECT * FROM nv_tab'
    result = connection.execute(sql).fetchall()

    result = {entry[0]: entry[1] for entry in result}


    connection.close()

    return (result, 200)


def main (request):

    engine = connect_to_cloud_sql()
    connection = engine.connect()
    sql = 'CREATE TABLE IF NOT EXISTS nv_tab (id INT NOT NULL AUTO_INCREMENT, val VARCHAR(255), PRIMARY KEY (id))'
    connection.execute(sql)
    connection.close()

    if request.method == 'POST':
        return fun_post(request)
    elif request.method == 'GET':
        return get_text_entries()