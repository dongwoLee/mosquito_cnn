import csv
import calendar
import datetime
import pandas as pd
import sys

standardDate=datetime.date(2012,5,7)
lastDate = datetime.date(2012,10,14)

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
 
data = pd.read_csv('Salesio_change_1.csv',skipfooter=1)#skipfooter=1
mos = (data[str(standardDate)].tolist())
mos = mos[-1]

wholeTempList = []

for i in range(len(wholeDateList)):
    temp = list(set(wholeDateList[i]) & set(wholeDate))
    temp =  (sorted(temp,key=lambda d:tuple(map(int,d.split('-')))))
    temp = temp[::-1] #temp_date reverse
    wholeTempList.append(temp)



for j in range(len(wholeTempList)):
    result = []
    for k in range(len(wholeTempList[j])):
        temp_data = pd.DataFrame(data[wholeTempList[j][k]])
        result.append(temp_data)
    result = pd.concat(result, axis=1)
    result.index = ['hum','hum5','hum10','hum15','hum20','hum25','hum30','raf','raf5','raf10','raf15','raf20','raf25','raf30','rfd',
                    'rfd5','rfd10','rfd15','rfd20','rfd25','rfd30','tav','tav5','tav10','tav15','tav20','tav25','tav30',',tmi','tmi5',
                    'tmi10','tmi15','tmi20','tmi25','tmi30','tmx','tmx5','tmx10','tmx15','tmx20','tmx25','tmx30','landuse']
    result.to_csv('C://Users/dw/Desktop/mosquito_cnn/Location_allDate/Salesio/'+str(wholeTempList[j][0])+'.csv')
    


















