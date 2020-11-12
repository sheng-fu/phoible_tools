import re
import csv
import itertools
from scipy.stats.stats import pearsonr
import numpy as np
import random
import pandas as pd
from scipy import stats
from collections import Counter
import operator
import copy
import urllib.request

def read_phoible(path = "https://raw.githubusercontent.com/phoible/dev/master/data/phoible.csv"):
    """
    Function that reads phoible.csv and turn it into a list of rows
    :param path: path of the phoible data (in csv)
    :
    """
    response = urllib.request.urlopen(path)
    lines = [l.decode('utf-8') for l in response.readlines()]
    phoible = [x for x in csv.reader(lines)]

    return phoible

def make_var_to_index(phoible):
    """
    Function that makes gives a mapping between variable/column names and list index 
    for phoible data
    :param phoible: phoible data as a list of rows (which are lists)
    :return: a dictionary {variable name: list index}
    """

    var_to_index = {}
    count = 0
    for i in phoible[0]:
        var_to_index[i] = count 
        count += 1

    return var_to_index

def get_phoible_feature_list(var_to_index):
    """
    Function that takes a var_to_index object and return a list of Phoible segment features
    :param var_to_index: a dictionary mapping variable name to index(column) number in Phoible data
    :return :     
    """
    return list(var_to_index.keys())[11:]

def parse_phoible(phoible, var_to_index, p2f):
    """
    Function that parse phoible (as a list) into a dictionary
    :param phoible: phoible data as a list of rows
    :param var_to_index: a dictionary that maps variable/column names to list index in phoible data
    :return: a dictionary {inventory code: inventory-level information as a dictionary}
    """

    parsed_phoible = {}

    for temp in phoible[1:-1]:
        parsed_phoible[temp[var_to_index['InventoryID']]] = parsed_phoible.get(temp[var_to_index['InventoryID']], 
            {"phonemes":[], 
            "InventoryID": temp[var_to_index['InventoryID']], 
            "Glottocode": temp[var_to_index['Glottocode']], 
            "ISO6393": temp[var_to_index['ISO6393']], 
            "LanguageName": temp[var_to_index['LanguageName']], 
            "SpecificDialect": temp[var_to_index['SpecificDialect']],
                "GlyphID": temp[var_to_index['GlyphID']]})
        if temp[-1] != 'N' and temp[var_to_index['tone']] != '+':
            parsed_phoible[temp[var_to_index['InventoryID']]]['phonemes'].append(temp[var_to_index['Phoneme']])
        parsed_phoible[temp[var_to_index['InventoryID']]]['phonemes'] = list(set(parsed_phoible[temp[var_to_index['InventoryID']]]['phonemes']))

        parsed_phoible[temp[var_to_index['InventoryID']]]['vowels'] = [x for x in parsed_phoible[temp[var_to_index['InventoryID']]]['phonemes'] if '+' in p2f[x]['syllabic']]
        parsed_phoible[temp[var_to_index['InventoryID']]]['consonants'] = [x for x in parsed_phoible[temp[var_to_index['InventoryID']]]['phonemes'] if x not in parsed_phoible[temp[var_to_index['InventoryID']]]['vowels']]

        if len(parsed_phoible) == 3021:
            break

    return parsed_phoible

def make_p2f(phoible, var_to_index):
    p2f = {}
    p2bof = {}

    for temp in phoible[1:]:
        if temp[-1] != 'N':
            if temp[var_to_index['Phoneme']] not in p2f.keys():
                p2f[temp[var_to_index['Phoneme']]] = {}
                p2bof[temp[var_to_index['Phoneme']]] = []
                for i in list(var_to_index.keys())[11:]:
                    p2f[temp[var_to_index['Phoneme']]][i] = temp[var_to_index[i]]
                    p2bof[temp[var_to_index['Phoneme']]].append(temp[var_to_index[i]]+i)

    #add laminal?
    keys = list(p2f.keys())
    for key in keys:
        if '̻' in key:
            p2f[key]['laminal'] = '+'
            p2bof[key] = [value+key for key, value in p2f[key].items()]
        else:
            p2f[key]['laminal'] = '0'
            p2bof[key] = [value+key for key, value in p2f[key].items()]

  
    #anteriority
    keys = list(p2f.keys())
    for key in keys:
        if '̟' not in key and key+'̟' not in keys:
            temp = copy.copy(p2f[key])
            p2f[key+'̟'] = temp
            p2f[key+'̟']['anterior'] = '+'
            p2f[key+'̟']['front'] = '+'
            p2bof[key+'̟'] = [value+key for key, value in p2f[key+'̟'].items()]
  
    #palatalization
    keys = list(p2f.keys())
    for key in keys:
        if 'ʲ' not in key and key+'ʲ' not in keys:
            temp = copy.copy(p2f[key])
            p2f[key+'ʲ'] = temp
            p2f[key+'ʲ']['back'] = '-'
            p2f[key+'ʲ']['dorsal'] = '+'
            p2f[key+'ʲ']['front'] = '+'
            p2bof[key+'ʲ'] = [value+key for key, value in p2f[key+'ʲ'].items()]
        
    #long segments
    keys = list(p2f.keys())
    for key in keys:
        if 'ː' not in key and key+'ː' not in keys:
            temp = copy.copy(p2f[key])
            p2f[key+'ː'] = temp
            p2f[key+'ː']['long'] = '+'
            p2bof[key+'ː'] = [value+key for key, value in p2f[key+'ː'].items()]

    #print(p2f['s̻'])
    #print(p2f['ʃ'])
    #exit()
    
    return p2f, p2bof