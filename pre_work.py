import pandas as pd
import csv
from collections import OrderedDict

def csvRead(file):
    with open(file,'r') as f:
        reader = csv.reader(f)
        readRow = list(reader)

    return readRow

def classifyByRegion(regionName,rList):
    for i in range(len(rList)):
        if(rList[i][1]=='Yunjoong/ElementrySchool'):
            with open('Yunjoong_ElementrySchool.csv','a') as myFile:
                wr = csv.writer(myFile,quoting = csv.QUOTE_ALL)
                wr.writerow(rList[i])
                myFile.close()
            

if __name__ == '__main__':
    rowList = csvRead('/Users/leedongwoo/Desktop/Mosquito_CNN/mos_factor_process.csv')
    regionList = []
    data = pd.read_csv('/Users/leedongwoo/Desktop/Mosquito_CNN/mos_factor_process.csv',usecols=['Location'])

    regionList = data.drop_duplicates().values.tolist()
    print (regionList)
    classifyByRegion(regionList,rowList)


