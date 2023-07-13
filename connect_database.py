from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

cloud_config= {
'secure_connect_bundle': 'secure-connect-insurance.zip'
}



auth_provider = PlainTextAuthProvider('KnhDDzndWwoqLodoPCkLsdZO', 'abq-UUFB839oew1NlHs2RC5EZGJtHevysQtZw1AEFc18nYNUGUCysJgo3vnEbjbOHSX2Aryrx,NBs.CMnu_SFmZZ7DdDZwqzKfY5,HKHyR,iZDyDSTj5ewUGJrFsZFUB')
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

row = session.execute("select release_version from system.local").one()
if row:
    print(row[0])
else:
    print("An error occurred.")