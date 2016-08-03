import pandas as pd
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient()
db = client['megan']
coll = db['so']

# Read query into DataFrame
df = pd.DataFrame(list(coll.find({'Description': {'$exists': True}})))

# My scraper created many fields in my MongoDB that were either useless or empty
# Remove those and this _id field
bad_redtexts = ['redtext{}'.format(num) for num in xrange(3,16)]
df.drop(bad_redtexts + ['_id'], axis=1, inplace=True)

# Remove colons from the names of fields (Residual from format on web)
for column in df.columns:
    if ':' in column:
        df[column.strip(':')] = df[column]
        df.drop(column, axis=1, inplace=True)

# Remove SOs who were last reported as deported or are currently incarcerated
deported = df[df['redtext0'].str.contains('Deported')].index
incarcerated = df[df['redtext0'] == 'INCARCERATED'].index
df.drop(set(deported) | set(incarcerated), axis = 0, inplace = True)

# Get rid of unicode characters that can't be written into csv
def change_text(text):
    if type(text) == unicode:
        return text.encode('utf-8')
    else:
        return text
for column in df.columns:
    df[column] = df[column].apply(change_text)

# Change lists to strings of ; delimited values
df['Description'] = df['Description'].apply(lambda x: ';'.join(x))
df['Offense Code'] = df['Offense Code'].apply(lambda x: ';'.join(x))

# Write to a CSV
with open('data/Megans_List.csv', 'w') as f:
    df.to_csv(f, index=False)
