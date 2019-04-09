import pandas as pd
import numpy as np

# -----------------------------
# EPC building stock data
# -----------------------------

epc_filepath = 'https://media.githubusercontent.com/media/LondonEnergyMap/cleandata/master/epc/domestic/epc_ldnstockmodel.csv'

dfepc = pd.read_csv(epc_filepath)
df = dfepc.copy()

print('epc data read')

# map EPC band
epc_dict = {'A': 1, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
df['epc_band'] = df.curr_enr.map(epc_dict)

# 13 invalid epc entries => drop
df = df[df.epc_band.notnull()]

# -----------------------------
# map mainsgas to 1/0, and assume 1 for nan values
# except postcodes not on gas grid
notgas_filepath = 'https://raw.githubusercontent.com/LondonEnergyMap/rawdata/master/building_stock/pcode_notgas.csv'
dfnotgas = pd.read_csv(notgas_filepath)

dfnotgas['gas'] = 0
notgas_dict = pd.Series(dfnotgas.gas.values,
                        index=dfnotgas['Postcode']).to_dict()

df['newgas'] = df.pcode.map(notgas_dict)
df.newgas.fillna(1, inplace=True)
df['mainsgas1'] = np.where(df.mainsgas.isnull(), df.newgas, df.mainsgas)

mains_dict = {'Y': 1, 'N': 0, 1: 1, 0: 0}
df['maingas'] = df.mainsgas1.map(mains_dict)

# -----------------------------
# map nrooms=-1 to nroom based on floor area bands similar to NEED

# first drop tfa == 0
df = df[~(df.tfa == 0)]

# create area bins and convert to nrooms for entries of -1
areabins = np.arange(0, 5000, 50).tolist()
arealabels = range(len(areabins)-1)

df['nroomtfa'] = pd.cut(df.tfa, areabins, labels=arealabels)
df['nroom'] = np.where((df.nrooms == -1), df.nroomtfa, df.nrooms)

# -----------------------------
# convert wall and transaction type to building age

# create new column of building age based on wall description
# and transcation type
df['wall_firstword'] = df.wall.str.split().str.get(0)
wall_mapping = {'Cavity': 2, 'System': 3, 'Timber': 3}
df['age'] = df.wall_firstword.map(wall_mapping)
df.age.fillna(1, inplace=True)
df.loc[df.transact_type == 'new dwelling', 'age'] = 6

# -----------------------------
# convert property type and form to exposed sides

# create new column for number of exposed sides based on property type and form
prop_mapping = {'House': 0, 'Flat': -2, 'Bungalow': 0.5, 'Maisonette': -2,
                'Park home': 0}
built_mapping = {'Detached': 0, 'Semi-Detached': -1,
                 'End-Terrace': -1, 'Mid-Terrace': -2,
                 'Enclosed Mid-Terrace': -2.5, 'Enclosed End-Terrace': -1.5,
                 'NO DATA!': 0}

df['propmap'] = df.prop_type.map(prop_mapping)
df['builtmap'] = df.builtform.map(built_mapping)
df['exposedsides'] = 6 + df.propmap + df.builtmap

df['type'] = df.prop_type.str.lower()
df['form'] = df.builtform.str.lower()

# -----------------------------
lsoa_filepath = 'https://raw.githubusercontent.com/LondonEnergyMap/rawdata/master/building_stock/postcode_lsoa_ldn.csv'

dflsoa = pd.read_csv(lsoa_filepath)

# -----------------------------
# make lsoa - postcode lookup dict
lsoa_dict = pd.Series(dflsoa.lsoa11nm.values, index=dflsoa.pcds).to_dict()

df['lsoa'] = df.pcode.map(lsoa_dict)

# some postcodes not matched to lsoa
# split postcode to 2 parts
# match first part to most commonly occurring lsoa

dflsoa['pc1'] = dflsoa.pcds.str.split().str[0]

pc1_lsoa = dflsoa.groupby(['pc1', 'lsoa11nm'],
                          as_index=False).count()[['pc1', 'lsoa11nm', 'pcds']]
pc1_lsoa.sort_values(by=['pcds'], ascending=False, inplace=True)
pc1_lsoa.drop_duplicates(subset='pc1', keep='first')

# make pc1 - lsoa dictionary and match postcodes where lsoa is null

pc1_dict = pd.Series(pc1_lsoa.lsoa11nm.values, index=pc1_lsoa.pc1).to_dict()

df['pc1'] = df.pcode.str.split().str[0]
df['lsoa'] = np.where(df.lsoa.isnull(), df.pc1.map(pc1_dict), df.lsoa)

# still 81 null lsoa that didnot match which we should drop
df = df[df.lsoa.notnull()]

print(df.head())

# -----------------------------

imd_filepath = 'https://raw.githubusercontent.com/LondonEnergyMap/rawdata/master/building_stock/imd_lsoa.csv'

dfimd = pd.read_csv(imd_filepath)

# create imd column based on lsoa
dfimd.columns = ['lsoacd', 'lsoa', 'lacd', 'la', 'imdrank', 'imd']
imd_dict = pd.Series(dfimd.imd.values, index=dfimd.lsoa).to_dict()

df['imd'] = df.lsoa.map(imd_dict)

keep_cols = ['bref',
             'lsoa',
             'imd',
             'type',
             'form',
             'exposedsides',
             'tfa',
             'nroom',
             'maingas',
             'age',
             'epc_band'
             ]

df = df[keep_cols]

# -----------------------------
# rename property types and forms in epc stock
# to match aggregated data in VOA dataset

df['property'] = df.form + " " + df.type


df['property'] = np.where(df.property.str.contains('flat'),
                          'flat', df.property)
df['property'] = np.where(df.property.str.contains('maisonette'),
                          'flat', df.property)
df['property'] = np.where(df.property.str.contains('bungalow'),
                          'bungalow', df.property)
df['property'] = np.where(df.property.str.contains('terrace house'),
                          'terraced house', df.property)
df['property'] = np.where(df.property.str.contains('semi'),
                          'semi house', df.property)
df['property'] = np.where(df.property.str.contains('detached'),
                          'detached house', df.property)

properties = ['bungalow', 'flat', 'detached house',
              'semi house', 'terraced house']

mask = df.property.isin(properties)
df.loc[~mask, 'property'] = 'other_unknown'

# -----------------------------

print(df.head())
df.to_csv('epc_lsoastockmodel.csv', index=False)
