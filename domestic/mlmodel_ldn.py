import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# filepath = 'https://raw.githubusercontent.com/LondonEnergyMap/model_data/master/domestic/combined_needepc.csv'

filepath = 'combined_needepc.csv'
df_all = pd.read_csv(filepath)

# ---------------------------
# mlmodel based on epc (and need?)
# ---------------------------

# choose epc entries only

temp = df_all[(df_all.gmeters > 1) & (df_all.emeters > 1)]

x = temp[['tfa', 'epc', 'age', 'nroom', 'exposedsides', 'maingas', 'imd']]
y = temp[['gcons', 'econs']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)

lrmodel = LinearRegression()
lrmodel.fit(x_train, y_train)
predictions = lrmodel.predict(x_test)
scores = lrmodel.score(x_train, y_train)

print('model score is ' + str(scores))

# ---------------------------
# apply model to london building stock
# ---------------------------

# ---------------------------
# VOA building stock data by type and nbedrooms
voastockpath = 'voatype_lsoastock.csv'
df_voastock = pd.read_csv(voastockpath)

# total dwellings for london
ndwellings_ldn = df_voastock.ndwellings.sum()
print('total dwellings in london ' + str(ndwellings_ldn))

# ---------------------------
# count epc stock and append up to total dwellings in london
epcstockpath = 'epc_lsoastock.csv'
df_epcstock = pd.read_csv(epcstockpath)

# count how many dwellings in epc
ndwellings_epc = df_epcstock.bref.count()
print('EPC buildig stock ' + str(ndwellings_epc))

# count how many more dwellings required to represent london
missing = ndwellings_ldn - ndwellings_epc

# calculate difference and randomly sample dwellings from epc and append to stock
repeat_epcstock = df_epcstock.sample(n=missing, random_state=42)
df_epcstock = df_epcstock.append(repeat_epcstock)

# ---------------------------

x_ldn = df_epcstock[['tfa', 'epc_band', 'age', 'nroom', 'exposedsides', 'maingas', 'imd']]

ldnpredictions = lrmodel.predict(x_ldn)
gcons_predict, econs_predict = map(list, zip(*ldnpredictions))

# ---------------------------
df = df_epcstock.copy()

df['gcons_predict'] = gcons_predict
df['econs_predict'] = econs_predict

gconsldn = df.gcons_predict.sum()
econsldn = df.econs_predict.sum()

# ---------------------------
# crosscheck with london consumption stats
# ---------------------------

gaspath = 'https://media.githubusercontent.com/media/LondonEnergyMap/cleandata/master/consumption/gasldn2015.csv'
elecpath = 'https://media.githubusercontent.com/media/LondonEnergyMap/cleandata/master/consumption/elecldn2015.csv'

dfgas = pd.read_csv(gaspath)
dfelec = pd.read_csv(elecpath)

# convert to numeric type and turn missing values '-' to NaNs (only in gas)
dfgas['gas'] = pd.to_numeric(dfgas.gas, errors='coerce')
dfelec['gas'] = pd.to_numeric(dfelec.elec)

# NOTE: compared to sub-national statistics,
# the boroughs with missing values did not give lower total consumption
# this suggests that the data is simply missing

# sort by lsoa code then interpolate missing values using nearest value
dfgas.sort_values(by='lsoa', inplace=True)
dfgas['gas'] = dfgas.gas.interpolate(method='nearest')

gasldn = dfgas.gas.sum()
elecldn = dfelec.elec.sum()

# ---------------------------
# print results
# ---------------------------

print('predict gas consumption is ' + str(round(gconsldn/1000000)) + ' MWh')
print('Actual gas consumption is ' + str(round(gasldn/1000000)) + ' MWh')

print('predict electricity consumption is ' + str(round(econsldn/1000000)) + ' MWh')
print('Actual electricity consumption is ' + str(round(elecldn/1000000)) + ' MWh')
