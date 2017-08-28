import csv
import calendar
import datetime
import pandas as pd
import sys

standardDate=datetime.date(2011,6,1)
lastDate = datetime.date(2011,10,30)

daterange = pd.date_range(standardDate,lastDate)

wholeDate = [] # just date
wholeDateList = [] #31 accumulation list

with open('Salesio_change_1.csv','r') as f:
    reader = csv.reader(f,delimiter=',')
    wholeDate =  next(reader) # whole date data

for single_date in daterange:
    dateList = []
    for i in range(0,31):
        end_date = single_date+datetime.timedelta(days=-i)
        end_date = str(end_date)[0:10]
        (dateList.append(str(end_date)))
    wholeDateList.append(dateList) #make wholeDate List 


data = pd.read_csv('Salesio_change_1.csv')
mos = (data[str(standardDate)].tolist())
mos = mos[-1]

















