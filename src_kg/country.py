import pycountry
import pandas as pd


mp = {"Laos":"Lao People's Democratic Republic",
      'Democratic Republic of the Congo':"Congo",
      'Palestine State':'The State of Palestine'}





df = pd.read_csv("../out/Country.csv")

COUNTRY_NAMES = df["country"].to_list()

# ~ COUNTRY_NAMES = [mp[c] if c in mp.keys() else c for c in COUNTRY_NAMES ]

COUNTRY = dict()
for c in COUNTRY_NAMES:
    if c in mp.keys():
        cc = mp[c]
    else:
        cc = c
    try:
        alc = pycountry.countries.get(name=cc).alpha_3
        # ~ print(pycountry.countries.get(name=c).alpha_2)
    except:
        alc = pycountry.countries.search_fuzzy(cc)[0].alpha_3
        # ~ print(pycountry.countries.search_fuzzy(c)[0].alpha_2)
    COUNTRY[c]=alc
