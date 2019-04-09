import pandas as pd
import numpy as np

file_type = 'https://raw.githubusercontent.com/LondonEnergyMap/rawdata/master/building_stock/dwelling-property-type-2015-lsoa-msoa.csv'

dftype = pd.read_csv(file_type)

# ------------------

# use LSOA geography
dftype_lsoa = dftype[dftype['GEOGRAPHY'].str.contains('LSOA')]
dftype_lsoa = dftype_lsoa.drop(columns='GEOGRAPHY')

typecols = dftype_lsoa.columns

ldnstock = pd.melt(dftype_lsoa, id_vars=typecols[0:3], value_vars=typecols[4:36],
                   var_name='prop_type', value_name='ndwellings')

# ------------------
# drop rows which contains 'all' property types
droprows = ['BUNGALOW',
            'FLAT_MAIS',
            'HOUSE_TERRACED',
            'HOUSE_SEMI',
            'HOUSE_DETACHED',
            'ALL_PROPERTIES']
ldnstock = ldnstock[(~ldnstock.prop_type.isin(droprows))]

# extract number of bedroom from entry
ldnstock['nbeds'] = ldnstock.prop_type.str.extract('(\d+)')

# convert column to lower case
ldnstock['prop_type'] = ldnstock.prop_type.str.lower()

# ------------------
# split information of property type and form into 2 columns
types = ['bungalow', 'house', 'flat']
forms = ['terraced', 'semi', 'detached']
misc = ['other', 'unknown', 'annexe']

ldnstock['prop'] = ldnstock.prop_type
ldnstock['form'] = 'default'

for i in range(3):
    ldnstock['prop'] = np.where((ldnstock.prop_type.str.contains(types[i])),
                                types[i], ldnstock.prop)
    ldnstock['prop'] = np.where((ldnstock.prop_type.str.contains(misc[i])),
                                'other_unknown', ldnstock.prop)
    ldnstock['form'] = np.where(ldnstock.prop_type.str.contains(forms[i]),
                                forms[i], ldnstock.form)

ldnstock.drop(columns=['prop_type'], inplace=True)
ldnstock.columns = ldnstock.columns.str.lower()

# ------------------
# keep only entries with 'ALL' council tax band + drop column
ldnstock = ldnstock[(ldnstock.band.str.contains('All'))]
ldnstock.drop(columns=['band'], inplace=True)

# ------------------
# fillna for nbeds
ldnstock.nbeds.fillna(-1, inplace=True)

# ------------------
# convert ndwellings to numeric for summing
ldnstock['ndwellings'] = ldnstock.ndwellings.str.replace('-', '0')
ldnstock['ndwellings'] = pd.to_numeric(ldnstock.ndwellings)

# ------------------
# create property column that combines form and type
ldnstock['property'] = ldnstock.form + " " + ldnstock.prop
ldnstock['property'] = ldnstock.property.str.replace('default ', '')

# ------------------
# create pivot table of lsoa properties
lsoastock = ldnstock.groupby(['area_name', 'property', 'nbeds'],
                             as_index=False).sum()[['area_name', 'property', 'nbeds', 'ndwellings']]
lsoastock = lsoastock[lsoastock.ndwellings > 0]

lsoastock.to_csv('VOA_stock.csv', index=False)
