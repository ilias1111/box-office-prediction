from google.oauth2 import service_account
# import pymongo


# Service account for google cloud from json file
def get_gbq_credentials():
    credentials = service_account.Credentials.from_service_account_file(
        "Code/data_pipeline/big_query_credentials.json"
    )
    return credentials, credentials.project_id


# def set_up_mongo_connection():
#     # load json file as dictionary
#     with open('./mongo_cred.json') as f:
#         mongo_credentials = json.load(f)

#     client = pymongo.MongoClient('mongodb://{}:{}@{}:{}/'.format(mongo_credentials['username'],
#                                                                  mongo_credentials['password'],
#                                                                  mongo_credentials['host'],
#                                                                  mongo_credentials['port']), maxPoolSize=None)
#     mydb = client[mongo_credentials['database']]

#     return mydb
