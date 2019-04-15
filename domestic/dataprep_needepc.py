import pandas as pd
import numpy as np


# function to randomly generate numbers for floor area band
def areamap(areaband):
    return np.random.randint(*tfa_dict[areaband])


# final columns for both dataframes
rename_cols = ['hid',
               'imd',
               'type',
               'form',
               'exposedsides',
               'tfa',
               'epc',
               'age',
               'nroom',
               'maingas',
               'gcons',
               'econs',
               'gmeters',
               'emeters']

# ----------------------------------
# need database file
need_filepath = 'https://media.githubusercontent.com/media/LondonEnergyMap/cleandata/master/need/need_ldn.csv'
dfneed = pd.read_csv(need_filepath)

# select only year 2012 and drop colum
dfneed = dfneed[dfneed.year == 2012]
dfneed.drop(columns=['year'], inplace=True)

prop_map = {101: 'detached house',
            102: 'semi-detached house',
            103: 'end-terrace house',
            104: 'mid-terrace house',
            105: 'default bungalow',
            106: 'default flat'}

temp = dfneed.proptype.map(prop_map)
new = temp.str.split(' ', n=1, expand=True)
dfneed['type'] = new[1]
dfneed['form'] = new[0]

# create nrooms as floorarea band
dfneed['nroom'] = dfneed.floorarea_band

# create exposed sides column
type_mapping = {'house': 0, 'flat': -2, 'bungalow': 0.5}
form_mapping = {'detached': 0, 'semi-detached': -1, 'end-terrace': -1, 'mid-terrace': -2, 'default': 0}

dfneed['propmap'] = dfneed.type.map(type_mapping)
dfneed['formmap'] = dfneed.form.map(form_mapping)
dfneed['exposedsides'] = 6 + dfneed.propmap + dfneed.formmap

# convert age column to 1,2,3,4
dfneed['age'] = dfneed.age - 100

# randomly map floor area band as floor area
tfa_dict = {1: [30, 50], 2: [51, 100], 3: [101, 150], 4: [151, 200]}

# randomly assign floor area according to bands
dfneed['tfa'] = dfneed.floorarea_band.map(areamap)

# create gas and elec meters columns
dfneed['gasmeters'] = 1
dfneed['elecmeters'] = 1

need_cols = ['hid',
             'imd_eng',
             'type',
             'form',
             'exposedsides',
             'tfa',
             'epc_band',
             'age',
             'nroom',
             'mainheatfuel',
             'gcons',
             'econs',
             'gasmeters',
             'elecmeters']

dfneed = dfneed[need_cols]

print(dfneed.head())
# ----------------------------------

# epc database file based on 10 meters
epc_filepath = 'https://media.githubusercontent.com/media/LondonEnergyMap/cleandata/master/epc/domestic/epcdom_10meters.csv'
dfepc = pd.read_csv(epc_filepath)

# imd file lookup according to postcode
imd_filepath = 'https://media.githubusercontent.com/media/LondonEnergyMap/cleandata/master/epc/domestic/imd/ecp10meters_pcode_imd.csv'
dfimd = pd.read_csv(imd_filepath)

# map postcode to deprivation index
imd_dict = pd.Series(dfimd['Index of Multiple Deprivation Decile'].values, index=dfimd['Postcode']).to_dict()
dfepc['imd'] = dfepc.pcode.map(imd_dict)

# first limit nrooms upper limit and area upper & lower limits
n = 10
dfepc = dfepc[dfepc.nrooms <= n]

tfa_upper = n*50
tfa_lower = 30
dfepc = dfepc[(dfepc.tfa >= tfa_lower) & (dfepc.tfa <= tfa_upper)]

# map nrooms=-1 to nroom based on floor area bands similar to NEED
areabins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
arealabels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

dfepc['nroomtfa'] = pd.cut(dfepc.tfa, areabins, labels=arealabels)
dfepc['nroom'] = np.where(dfepc.nrooms == -1, dfepc.nroomtfa, dfepc.nrooms)


# create new column for number of exposed sides based on property type and form
prop_mapping = {'House': 0, 'Flat': -2, 'Bungalow': 0.5, 'Maisonette': -2,
                'Park home': 0}
built_mapping = {'Detached': 0, 'Semi-Detached': -1,
                 'End-Terrace': -1, 'Mid-Terrace': -2,
                 'Enclosed Mid-Terrace': -2.5, 'Enclosed End-Terrace': -1.5,
                 'NO DATA!': 0}

dfepc['propmap'] = dfepc.prop_type.map(prop_mapping)
dfepc['builtmap'] = dfepc.builtform.map(built_mapping)
dfepc['exposedsides'] = 6 + dfepc.propmap + dfepc.builtmap

dfepc['type'] = dfepc.prop_type.str.lower()
dfepc['form'] = dfepc.builtform.str.lower()

# create new column of building age based on wall description and transcation type
dfepc['wall_firstword'] = dfepc.wall.str.split().str.get(0)
wall_mapping = {'Cavity': 2, 'System': 3, 'Timber': 3}
dfepc['age'] = dfepc.wall_firstword.map(wall_mapping)
dfepc.age.fillna(1, inplace=True)
dfepc.loc[dfepc.transact_type == 'new dwelling', 'age'] = 6

# map mainsgas to 1/0, after filling in null values according to not on gas grid postcodes
notgas_filepath = 'https://raw.githubusercontent.com/LondonEnergyMap/rawdata/master/building_stock/pcode_notgas.csv'
dfnotgas = pd.read_csv(notgas_filepath)
dfnotgas['gas'] = 0
notgas_dict = pd.Series(dfnotgas.gas.values, index=dfnotgas['Postcode']).to_dict()
dfepc['newgas'] = dfepc.pcode.map(notgas_dict)
dfepc.newgas.fillna(1, inplace=True)
dfepc['mainsgas1'] = np.where(dfepc.mainsgas.isnull(), dfepc.newgas, dfepc.mainsgas)
mains_dict = {'Y': 1, 'N': 0, 1: 1, 0: 0}
dfepc['maingas'] = dfepc.mainsgas1.map(mains_dict)

# map epc bands to 1-6
epc_dict = {'A': 1, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
dfepc['epc_band'] = dfepc.curr_enr.map(epc_dict)

# keep only selected columns
epc_cols = ['bref',
            'imd',
            'type',
            'form',
            'exposedsides',
            'tfa',
            'epc_band',
            'age',
            'nroom',
            'maingas',
            'gasmid',
            'elecmid',
            'gasmeters',
            'elecmeters']

dfepc = dfepc[epc_cols]
print(dfepc.head())
# ----------------------------------

dfneed.columns = rename_cols
dfepc.columns = rename_cols
df = dfepc.append(dfneed)

df.to_csv('combined_needepc.csv', index=False)
