import json
import boto3
import pymysql




def handler(event, context):
    data = [
        {"CustID": 1054, "Name": "Richard Roe"},
        {"CustID": 121, "Name": "selen Roe"},
        {"CustID": 2, "Name": "Alex Valette"},
        {"CustID": 3, "Name": "Nico Vallot"}
    ]

    rds = boto3.client('rds')
    response = rds.describe_db_instances(DBInstanceIdentifier='rdsnicov')
    rds_endpoint = response['DBInstances'][0]['Endpoint']['Address']

    connection = pymysql.connect(
        host=rds_endpoint,
        user='admin',
        password='admin123',
        database='mydb'
    )

    with connection.cursor() as cursor:
        # Vérifier si la table existe
        cursor.execute("SHOW TABLES LIKE 'table'")
        result = cursor.fetchone()

        # Si la table n'existe pas, la créer
        if not result:
            cursor.execute("CREATE TABLE `table` (`CustID` INT, `Name` VARCHAR(255))")
            connection.commit()

        # Insérer les données dans la table
        for item in data:
            cursor.execute("INSERT INTO `table` (`CustID`, `Name`) VALUES (%s, %s)", (item['CustID'], item['Name']))
        connection.commit()

    connection.close()
    s3 = boto3.client('s3')
    s3.put_object(Bucket='bucketnicov', Key='data.json', Body=json.dumps(data))
    print("endoooo")