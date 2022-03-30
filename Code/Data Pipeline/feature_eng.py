import pandas as pd
import numpy as np
from math import sin, cos, pi
import cpi

certs = {"0": "G",
         "6": "G",
         "G": "G",
         "R": "R",
         "12": "PG13",
         "16": "PG16",
         "18": "R",
         "PG": "PG",
         "NC-17": "R",
         "PG-13": "PG13",
         "TV-14": "PG16",
         "Unrated": "U",
         "(Banned)": "R",
         "Not Rated": "U",
         "BPjM Restricted": "R"}


def fix_certificates(df):
    df['cert'] = df['cert'].replace(certs).fillna("U")
    return df


def adjust_cpi(amount, date):
    if date < 2018:

        try:
            return cpi.inflate(amount, date)
        except:
            return -1

    else:
        return amount


def sin_cos(n, k):
    theta = (2 * pi * n) / k
    return sin(theta), cos(theta)


def classify(s):
    if s < 0:
        return 'Bankrupt'
    elif s < 1_000_000:
        return 'Flop'
    elif s < 15_000_000:
        return "Small Movie"
    elif s < 50_000_000:
        return "Blockbuster"
    else:
        return "Success"


def classify_b(s):
    if s < 0:
        return 'Bankrupt'
    else:
        return "Success"
