# -*- coding: utf-8 -*-
"""
Created on Thu Apr 7 17:33:55 2022

@author: Manuel Rubio

Source:
    herb https://www.fao.org/faostat/es/#data/RP
    prod https://www.fao.org/faostat/es/#data/QCL
"""
import pandas as pd
import os
import seaborn as sb
import matplotlib.pyplot as plt

path_0 = os.path.join("C:\\Users\\33695\\OneDrive - UniLaSalle\\Documents\\Unilasalle\\Survey Methods\\Data Mining")
os.chdir(path_0)

# Open the csv file
herb = pd.read_csv("herbicides.csv") #tons of herb used per year
rto = pd.read_csv("yield.csv") #file with yield (kg/ha), area (ha) and productivity (tns)
fung_ins = pd.read_csv("fung+ins.csv")

area = rto[rto['Elemento'] == 'Área cosechada']

# select only yield rows
rto = rto[rto['Elemento'] == 'Rendimiento']
# transform hg to tn
rto['Valor'] = rto.Valor / 10000

# get one column for fung and other for ins
fung_ins = fung_ins.pivot(index=["Área", "Año"], columns='Producto', values='Valor')

# keep only useful columns
herb = herb[['Área', 'Año', 'Valor']]
rto = rto[['Área', 'Año', 'Valor']]
area = area[['Área', 'Año', 'Valor']]

# group by country and year
rto_group = rto.groupby(["Área", "Año"]).agg({"Valor": ['mean']})
rto_group.columns = ['Mean yield (tn/ha)']
rto_group = rto_group.rename_axis(["Área", "Año"]).reset_index()

# same for area
area_group = area.groupby(["Área", "Año"]).agg({"Valor": ['sum']})
area_group.columns = ['Sum area (ha)']
area_group = area_group.rename_axis(["Área", "Año"]).reset_index()

# merge dataframes
df_final = pd.merge(rto_group, herb, how ='left', left_on=['Área','Año'], right_on = ['Área','Año'])
df_final = pd.merge(df_final, fung_ins, how ='left', left_on=['Área','Año'], right_on = ['Área','Año'])
df_final = pd.merge(df_final, area_group, how ='left', left_on=['Área','Año'], right_on = ['Área','Año'])

# Rename columns
df_final = df_final.rename(columns={"Área": "Country", "Valor": "Herbicides (kg/ha)", \
                         "Fungicidas y bactericidas": "Fungicides (kg/ha)", \
                         "Insecticidas": "Insecticides (kg/ha)", 'Año': 'Year'})

# add GMO info
dict_GMO = {"Argentina" : "Yes", "Brasil" : "Yes", "Estados Unidos de América" : "Yes", \
            "Francia": "No", "Italia": "No", "Japón":"No"}
list_GMO = []
for country in df_final['Country']:
    list_GMO.append(dict_GMO.get(country))
df_final["GMO"] = list_GMO

# divide by surface to get kg/ha
df_final['Herbicides (kg/ha)'] = (df_final['Herbicides (kg/ha)'] / df_final['Sum area (ha)']) * 1000
df_final['Fungicides (kg/ha)'] = (df_final['Fungicides (kg/ha)'] / df_final['Sum area (ha)']) * 1000
df_final['Insecticides (kg/ha)'] = (df_final['Insecticides (kg/ha)'] / df_final['Sum area (ha)']) * 1000

# Correlation matrix
df_corr = df_final[['Mean yield (tn/ha)','Herbicides (kg/ha)','Fungicides (kg/ha)','Insecticides (kg/ha)']]
corr = df_corr.corr()
sb.heatmap(corr, annot=True)
plt.title("Correlation matrix (All data)")
plt.show()
# # Correlation matrix for GMO countries
# corr = df_final[df_final['GMO'] == 'Yes'].drop(['Year','Sum area (ha)'], axis = 1).corr()
# sb.heatmap(corr, annot=True)
# plt.title("Correlation matrix (GMO)")
# plt.show()
# # Correlation matrix for no-GMO countries
# corr = df_final[df_final['GMO'] == 'No'].drop(['Year','Sum area (ha)'], axis = 1).corr()
# sb.heatmap(corr, annot=True)
# plt.title("Correlation matrix (no-GMO)")
# plt.show()

#%% Supply evolution
# Change country names to english
dict_en = {"Argentina" : "Argentina", "Brasil" : "Brazil", "Estados Unidos de América" : "United States of America", \
            "Francia": "France", "Italia": "Italy", "Japón":"Japan"}
df_final = df_final.replace(dict_en)

# list to iterate coutrnies
country_list = list(df_final.Country.value_counts().reset_index()["index"])

# plot matrix 2x3
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
# list to iterate among plots
ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]

for countries, axes in zip(country_list, ax_list) :
    df_temp = df_final[df_final['Country'] == countries]
    x, y, y2, y3, y4, y5 = pd.to_datetime(df_temp['Year'], format="%Y"), df_temp['Mean yield (tn/ha)'], \
        df_temp['Herbicides (kg/ha)'] + df_temp['Fungicides (kg/ha)'] + df_temp['Insecticides (kg/ha)'], \
        df_temp['Herbicides (kg/ha)'], df_temp['Fungicides (kg/ha)'], df_temp['Insecticides (kg/ha)']
    axes.plot(x, y, linewidth='2', color='green', label= "Mean yield (tn/ha)")
    axes.plot(x, y2, linewidth='2', color='red', label= "Pesticides (kg/ha)")
    axes.plot(x, y3, linewidth='1', color='lightgreen', label= "Herbicides (kg/ha)")
    axes.plot(x, y4, linewidth='1', color='blue', label= "Fungicides (kg/ha)")
    axes.plot(x, y5, linewidth='1', color='orange', label= "Insecticides (kg/ha)")
    
    fig.suptitle('Yield and pesticides evolution')
    axes.set_title(countries)    
    plt.legend(loc = (1.04,0.9))
    plt.title(countries)

for ax in fig.get_axes():
    ax.tick_params(labelrotation=45)
    ax.label_outer()
plt.show()

#%% PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# create the array
X = df_final[['Mean yield (tn/ha)', 'Herbicides (kg/ha)', 'Fungicides (kg/ha)', 'Insecticides (kg/ha)']].to_numpy()

# Scale
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=4)
pca.fit(X)
print(pca.explained_variance_ratio_)

# pca to df 
x_pca = pca.fit_transform(X)
pca_df = pd.DataFrame(data=x_pca, columns=['pc1', 'pc2', 'pc3', 'pc4'])

# concatenate df
df_conc = pd.concat((df_final, pca_df), axis=1)

# Plot
sb.relplot(data=df_conc, x='pc1', y='pc3', hue='Country', style = "GMO", aspect=1.61)
plt.title("PCA")
plt.xlabel(f'Dim 1 | {(pca.explained_variance_ratio_[0] * 100).round()}%')
plt.ylabel(f'Dim 3 | {(pca.explained_variance_ratio_[2] * 100).round()}%')
plt.show()
