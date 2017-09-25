import pandas as pd

def changeToLevel(file):
	df = pd.read_csv(file)
	DateData =df['Date']
	mosData = df['Mosq']

	mosData = mosData.values.tolist()
	DateData = DateData.values.tolist()

	for i in range(len(mosData)):
		if(0<= mosData[i] and mosData[i]<=20):
			mosData[i]=1
		elif(21<= mosData[i] and mosData[i]<=40):
			mosData[i]=2
		elif(41<= mosData[i] and mosData[i]<=80):
			mosData[i]=3
		elif(81<= mosData[i] and mosData[i]<=160):
			mosData[i]=4
		elif(161<= mosData[i] and mosData[i]<=320):
			mosData[i]=5
		elif(321<= mosData[i] and mosData[i]<=640):
			mosData[i]=6
		elif(641<= mosData[i] and mosData[i]<=1280):
			mosData[i]=7
		elif(mosData[i] >= 1281):
			mosData[i]=8

	df_csv = pd.DataFrame(mosData)
	df_csv.to_csv("C:/Users/dw/Desktop/mosquito_cnn/Label_Data/Level/Yoonjung_Elementary_level.csv")


if __name__ == '__main__':
	changeToLevel("C:/Users/dw/Desktop/mosquito_cnn/Label_Data/Count_mosquito/Yoonjung_Elementary.csv")