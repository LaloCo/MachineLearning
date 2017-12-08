#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle, os, math
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
enron_data = pickle.load(open(os.path.dirname(CURRENT_DIR) + "/final_project/final_project_dataset_unix.pkl", "rb"))

pois_txt = open(os.path.dirname(CURRENT_DIR) + "/final_project/poi_names.txt", "rb")

## Rows
# print(len(enron_data))

## Columns
# print(len(enron_data["SKILLING JEFFREY K"].keys()))

## POIs
# pois = {p: v for p, v in enron_data.items() if v["poi"] == 1}
# print(len(pois))

## POIs in file
# pois = []
# for ln in pois_txt:
#     if ln.startswith(b"("):
#         pois.append(ln)
# print(len(pois))

## James Prentice's stock
# stock = enron_data["PRENTICE JAMES"]["total_stock_value"]
# print(stock)

## Stock options exercised by Jeffrey K Skilling
# emails = enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
# print(emails)

## CEO, Chariman & CFO total payments
# ceo = enron_data["SKILLING JEFFREY K"]["total_payments"]
# chairman = enron_data["LAY KENNETH L"]["total_payments"]
# cfo = enron_data["FASTOW ANDREW S"]["total_payments"]
# print(ceo)
# print(chairman)
# print(cfo)

## Salary and known email address
# salaries = {p: v for p, v in enron_data.items() if v["salary"] != 'NaN'}
# email_addressess = {p: v for p, v in enron_data.items() if v["email_address"] != 'NaN'}
# print(len(salaries))
# print(len(email_addressess))

## % of people w/o total payments
# total_payments = {p: v for p, v in enron_data.items() if v["total_payments"] == 'NaN'}
# print(len(total_payments) / len(enron_data))

## % of POIs w/o total payments
# pois = {p: v for p, v in enron_data.items() if v["poi"] == 1}
# total_payments = {p: v for p, v in pois.items() if v["total_payments"] == 'NaN'} # notice the filter happen from pois instead of from enron_data
# print(len(total_payments) / len(pois))

## numbers + 10
total_payments = {p: v for p, v in enron_data.items() if v["total_payments"] == 'NaN'}
print(len(enron_data) + 10)
print(len(total_payments) + 10)