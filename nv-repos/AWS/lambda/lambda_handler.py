import json
import sys
import logging
import pymysql
import boto3

rds_host = "nvadatabase.c96r5xa5f7rh.eu-west-3.rds.amazonaws.com"
user_name = "nicolasvallot"
password = "nicolasvallot"
db_name = "mysql"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

try:
    conn = pymysql.connect(host=rds_host, user=user_name, passwd=password, db=db_name, connect_timeout=5)
except pymysql.MySQLError as e:
    logger.error("ERROR: Unexpected error: Could not connect to MySQL instance.")
    logger.error(e)
    sys.exit()

logger.info("SUCCESS: Connection to RDS MySQL instance succeeded")


def lambda_handler(event, context):
    """
    This function creates a new RDS database table and writes records to it
    """
    with conn.cursor() as cur:
        cur.execute(
            "create table if not exists Customer ( CustID  int NOT NULL, Name varchar(255) NOT NULL, PRIMARY KEY (CustID))")
    conn.commit()

    s3 = boto3.client('s3')

    print("event : ", event)
    file_obj = event["Records"][0]
    print("file_obj : ", file_obj)
    filename = str(file_obj['s3']['object']['key'])
    print("filename : ", filename)
    fileObj = s3.get_object(Bucket="nvabucket", Key=filename)
    print("fileObj : ", fileObj)
    file_content = fileObj["Body"].read().decode('utf-8')
    print(file_content)

    with conn.cursor() as cur:
        # message = event['Records'][0]['body']
        print("file_content : ", file_content)
        data = json.loads(file_content)
        print('len data: ', len(data))
        for i in range(len(data)):
            CustID = data[i]['CustID']
            print("CustID : ", CustID)
            Name = data[i]['Name']
            item_count = 0
            sql_string = f"insert into Customer (CustID, Name) values({CustID}, '{Name}')"
            cur.execute(sql_string)
            conn.commit()
        cur.execute("select * from Customer")
        logger.info("The following items have been added to the database:")
        for row in cur:
            item_count += 1
            logger.info(row)
    conn.commit()

    return "Added %d items to RDS MySQL table" % (item_count)
