import pandas as pd
import os, glob
import shutil

def getFileName(dir):
	os.chdir(dir)
	dateList = []
	for file in glob.glob("*.csv"):
		dateList.append(file[0:10])

	return dateList

def moveCsv(file,DList):#Dlist is WholeDataList
	locationData = pd.read_csv(file)
	locationData = locationData['Date']

	dst = "C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/KBS/"

	temp = set(locationData) & set(DList)
	temp = (sorted(temp,key=lambda d:tuple(map(int,d.split('-')))))


	for i in range(len(temp)):
		shutil.copy("C:/Users/dw/Desktop/mosquito_cnn/split_csv/"+temp[i]+".csv",dst)

if __name__ == '__main__':
	wholeDataList = getFileName("C:/Users/dw/Desktop/mosquito_cnn/split_csv")
	moveCsv("C:/Users/dw/Desktop/mosquito_cnn/mosquito_data/KBS.csv",wholeDataList)

