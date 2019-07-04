from dataProcessor import processDataSet, generateXY, countAccuracy, countProfit, calculateRMSE
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from numpy import array


def main():

    print("processing dataset")
    dataSet = processDataSet()
    print("generating X, Y")
    X, Y, priceDifference = generateXY(dataSet)
    X = array(X)
    Y = array(Y)
    priceDifference = array(priceDifference)
    regressor = SVR(kernel='rbf', C=1e3, gamma=0.1, tol=1e-8)

    counter = 1

    kf = KFold(n_splits=5)

    for train_index, test_index in kf.split(X):

        print("TRAIN:[", str(train_index[0]) + ", " + str(train_index[len(train_index)-1]), "]",
              "TEST:[", str(test_index[0]) + ", " + str(test_index[len(test_index)-1]), "]")

        xtrain, xtest = X[train_index], X[test_index]

        ytrain, ytest = Y[train_index], Y[test_index]
        print("fitting...")
        regressor.fit(xtrain, ytrain)
        pred = regressor.predict(xtest)


        res = countAccuracy(pred, ytest)
        profit = countProfit(pred, priceDifference[test_index])
        print ("Tacnost ",  " : ", res, "%")
        print("Profit ", " : ", profit)
        print ("RMSE ", " : ", calculateRMSE(pred, ytest))
        counter+=1
main()
