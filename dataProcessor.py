import json
import os
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from datetime import datetime, timedelta
keyWords = ["buy", "invest", "investing", "bought", "gain", "buying",
            "sell", "selling", "loss", "up",
            "increase", "down"]
def processDataSet():
    dataSet = {}
    #words = {}
    tweetsFolder = "data/tweet/preprocessed/"
    for subdir, dirs, files in os.walk(tweetsFolder):
        for company_name in dirs:
            dataSet[company_name] = {}
            #print (os.path.join(tweetsFolder, company_name))
            for subdir, dirs, files in os.walk(tweetsFolder + "/" + company_name):
                for file in files:
                    dataSet[company_name][file] = {}
                    with open(os.path.join(tweetsFolder, company_name, file), "r") as read_file:
                        for line in read_file:
                            data = json.loads(line)
                            for word in data["text"]:
                                #if (words.get(word) == None):
                                 #   words[word] = 1
                               # else:
                                #    words[word] += 1
                                if dataSet[company_name][file].get(word) == None:
                                    dataSet[company_name][file][word] = 1
                                else:
                                    dataSet[company_name][file][word] += 1
    #sorted_x = sorted(words.items(), key=lambda kv: kv[1])
    #import collections

    #sorted_dict = collections.OrderedDict(sorted_x)
    #for key, item in sorted_dict.items():
        #print (key, item)
    return dataSet
def generateXY(dataSet):
    X = []
    Y = []
    priceDifferences = []
    companies = list(dataSet.keys())
    companies.sort()
    for companyName in companies:
        print(companyName, " start index : ", len(X))
        prices = pd.read_csv("data/price/raw/" + companyName + ".csv", delimiter=',')
        for date, data in dataSet[companyName].items():
            x = []
            for keyWord in keyWords:
                x.append(0) if data.get(keyWord) is None else x.append(data[keyWord])
            stockGrowth, priceDiff = calculateStockGrowth( date, prices)
            if (stockGrowth is not None and checkIfZeros(x) == False):
                X.append(x)
                Y.append(stockGrowth)
                priceDifferences.append(priceDiff)


        print(companyName, " end index: ", len(X))
    tfidf = TfidfTransformer()  # by default norm = "l2"
    tfidf.fit(X)

    tf_idf_matrix = tfidf.transform(X)
    #printXY(X, Y)
    return tf_idf_matrix.todense(), Y, priceDifferences
    #return X, Y, priceDifferences, startDates, companyNames
def printXY(X, Y, pred, dates, companyNames):

    for i in range(len(X)):
        resStr = companyNames[i] + " " +dates[i] + " ["
        for j in range(len(X[i])):
           resStr += keyWords[j] + " : " + str(X[i][j]) + " ,"
        resStr += "] " + "result : [" + str(Y[i]) + "]" + " predict :" + "[" + str(pred[i]) + "]"
        print(resStr)
        print("\n")
def calculateStockGrowth(date, prices):
    dateStart = datetime.strptime(date, '%Y-%m-%d')
    dateEnd = dateStart + timedelta(days=2)
    priceStart = None
    priceEnd = None
    for row in prices.values:
        dateStockStr = row[0]
        dateStock = datetime.strptime(dateStockStr, '%Y-%m-%d')
        if (dateStock == dateStart):
            priceStart = float(row[1])
        if (dateStock == dateEnd and priceStart is not None):
            priceEnd = float(row[4])
            result = 1 if priceEnd - priceStart > 0 else -1
            return result, priceEnd-priceStart
        if (dateStock > dateEnd):
            return None, None

def countAccuracy(pred, test):
    count = 0
    for i in range(len(pred)):
        if ((pred[i] >= 0 and test[i] >= 0) or pred[i] <= 0 and test[i] <= 0):
            count += 1
    return (count/len(pred))*100
def countProfit(pred, priceDiff):
    sum = 0
    for i in range(len(pred)):
        if (pred[i] > 0):
            sum += priceDiff[i]
    return sum
def calculateRMSE(pred, actual):
    sum = 0
    for i in range(len(pred)):
        sum += pow(actual[i] - pred[i], 2)
    return sum/len(pred)
def checkIfZeros(X):
    for x in X:
        if x > 0:
            return False
    return True