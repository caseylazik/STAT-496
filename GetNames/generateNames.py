import pandas as pd

gender_by_name = pd.read_csv("Data\\name_gender_dataset.csv")  #https://archive.ics.uci.edu/dataset/591/gender+by+name
first_name_race_probs = pd.read_csv("Data\\first_nameRaceProbs.csv") #https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SGKW0K
last_names = pd.read_csv("Data\\last_raceNameProbs.csv") #https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SGKW0K

gender_by_name['name_norm'] = gender_by_name['Name'].str.upper()
first_name_race_probs['name_norm'] = first_name_race_probs['name'].str.upper()

first_names = gender_by_name.merge(first_name_race_probs, on='name_norm', how='inner')

first_names['male'] = 0
first_names['female'] = 0
for i in range(0, len(first_names)-1):
    if first_names.iloc[i]['Name'] == first_names.iloc[i-1]['Name']:
        total = first_names.iloc[i]['Count'] + first_names.iloc[i+1]['Count']
        if first_names.iloc[i]['Gender'] == 'F':
            first_names.loc[first_names.index[i], 'female'] = first_names.iloc[i]['Count'] / total
            first_names.loc[first_names.index[i], 'male'] = first_names.iloc[i-1]['Count'] / total
        elif first_names.iloc[i]['Gender'] == 'M':
            first_names.loc[first_names.index[i], 'male'] = first_names.iloc[i]['Count'] / total
            first_names.loc[first_names.index[i], 'female'] = first_names.iloc[i-1]['Count'] / total
    elif first_names.iloc[i]['Name'] == first_names.iloc[i+1]['Name']:
        total = first_names.iloc[i]['Count'] + first_names.iloc[i+1]['Count']
        if first_names.iloc[i]['Gender'] == 'M':
            first_names.loc[first_names.index[i], 'male'] = first_names.iloc[i]['Count'] / total
            first_names.loc[first_names.index[i], 'female'] = first_names.iloc[i+1]['Count'] / total
        elif first_names.iloc[i]['Gender'] == 'F':
            first_names.loc[first_names.index[i], 'female'] = first_names.iloc[i]['Count'] / total
            first_names.loc[first_names.index[i], 'male'] = first_names.iloc[i+1]['Count'] / total
    elif first_names.iloc[i]['Gender'] == 'M': 
        first_names.loc[first_names.index[i], 'male'] = 1
        first_names.loc[first_names.index[i], 'female'] = 0
    elif first_names.iloc[i]['Gender'] == 'F': 
        first_names.loc[first_names.index[i], 'male'] = 0
        first_names.loc[first_names.index[i], 'female'] = 1


wm_firstnames = first_names[(first_names['whi'] > 0.9) & (first_names['male'] > 0.9) & (first_names['Count'] > 20000)]
wf_firstnames = first_names[(first_names['whi'] > 0.9) & (first_names['female'] > 0.9) & (first_names['Count'] > 20000)]
bm_firstnames = first_names[(first_names['bla'] > 0.9) & (first_names['male'] > 0.9) & (first_names['Count'] > 10000)]
bf_firstnames = first_names[(first_names['bla'] > 0.9) & (first_names['female'] > 0.9) & (first_names['Count'] > 10000)]
hm_firstnames = first_names[(first_names['his'] > 0.8) & (first_names['male'] > 0.8) & (first_names['Count'] > 10000)]
hf_firstnames = first_names[(first_names['his'] > 0.8) & (first_names['female'] > 0.8) & (first_names['Count'] > 10000)]
am_firstnames = first_names[(first_names['asi'] > 0.8) & (first_names['male'] > 0.8) & (first_names['Count'] > 1000)]
af_firstnames = first_names[(first_names['asi'] > 0.8) & (first_names['female'] > 0.8) & (first_names['Count'] > 1000)]


last_names["prob"] = last_names["whi"] + last_names["bla"] + last_names["his"] + last_names["asi"] + last_names["oth"]
last_names["whi"] = last_names["whi"] / last_names["prob"]
last_names["bla"] = last_names["bla"] / last_names["prob"]
last_names["his"] = last_names["his"] / last_names["prob"]
last_names["asi"] = last_names["asi"] / last_names["prob"]

whi_lastnames = last_names[(last_names['whi'] > 0.6) & (last_names["prob"] > 0.0002)]
bla_lastnames = last_names[(last_names['bla'] > 0.75) & (last_names["prob"] > 0.00015)]
his_lastnames = last_names[(last_names['his'] > 0.9) & (last_names["prob"] > 0.0001)]
asi_lastnames = last_names[(last_names['asi'] > 0.9) & (last_names["prob"] > 0.0001)]


seed = 42


def sample_names(df, n=3):
    return df.sample(n=n, replace=False, random_state=seed)['name'].tolist()


wm = sample_names(wm_firstnames, 3)
wf = sample_names(wf_firstnames, 3)
bm = sample_names(bm_firstnames, 3)
bf = sample_names(bf_firstnames, 3)
hm = sample_names(hm_firstnames, 3)
hf = sample_names(hf_firstnames, 3)
am = sample_names(am_firstnames, 3)
af = sample_names(af_firstnames, 3)
wln = sample_names(whi_lastnames, 3)
bln = sample_names(bla_lastnames, 3)
hln = sample_names(his_lastnames, 3)
aln = sample_names(asi_lastnames, 3)

names = []

names += [f"{fn} {ln}" for fn, ln in zip(wm, wln)]
wln = sample_names(whi_lastnames, 3)
names += [f"{fn} {ln}" for fn, ln in zip(wf, wln)]
names += [f"{fn} {ln}" for fn, ln in zip(bm, bln)]
bln = sample_names(bla_lastnames, 3)
names += [f"{fn} {ln}" for fn, ln in zip(bf, bln)]
names += [f"{fn} {ln}" for fn, ln in zip(hm, hln)]
hln = sample_names(his_lastnames, 3)
names += [f"{fn} {ln}" for fn, ln in zip(hf, hln)]
names += [f"{fn} {ln}" for fn, ln in zip(am, aln)]
aln = sample_names(asi_lastnames, 3)
names += [f"{fn} {ln}" for fn, ln in zip(af, aln)]


with open("names.txt", "w") as f:
    f.write("\n".join(names))
