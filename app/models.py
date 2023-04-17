import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras import callbacks
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
import dill


#Preparing Data


#creating dataframe called "df", which reads "covtype.csv" file
df = pd.read_csv("covtype.data", sep=",", header=None)

#assigning features to "features" and target (cover type) to "target"
features = df.iloc[:, 1:-1]
target = df.iloc[:, -1:]

#splitting dataset to training dataset and testing dataset, with test size of 25% of entire dataset
xTrain, xTest, yTrain, yTest = train_test_split(features, target, test_size=0.25)

#creating scaler to scale features
scaler = StandardScaler()

#fitting and transforming "xTrain" with "yTrain" in a scaler
xTrainScaled = scaler.fit_transform(xTrain, yTrain)

#transforming "xTest" in a scaler, but not fitting it with "yTest", because we don't want biased data
xTestScaled = scaler.transform(xTest)


#Heuristic Algorithm


def heuristicPredict(dfPredict):
    #importing stuff in function to make  REST API work properly
    import pandas as pd
    df = pd.read_csv("covtype.data", sep=",", header=None)

    #creating new data frame that is df, but without name
    df1 = df.drop(0, axis="columns")

    #creating "dataDict" list to store average of azimuths of each covtype
    dataDict =  []

    #creating for loop to get average of azimuths of each covtype
    #for loop that goes through every covtype
    for covtype in sorted(df.loc[:, 54].unique()): 

        sum1 = 0
        numRows = 0

        #for loop that goes through every row
        for i in range(0, round(len(df1.index))):
            #checking if current row is current covtype
            if df.loc[i, 54] == covtype:
                #for loop that goes through every column
                for column in df1.columns:
                    #checking if column is 3 (azimuth)
                    if column == 3:
                        #adding azimuth to sum
                        sum1 += int(df1.loc[i, column])
                numRows += 1
        #appending covtype and its average of azimuths to "dataDict" dictionary
        dataDict.append({
            "covtype": covtype,
            "azimuth": sum1/numRows,
        })

    #getting azimuth from dfPredict
    azimuth = int(dfPredict.iloc[:, 1])

    #getting values of average azimuth of certain covtype
    covtype1Azimuth = dataDict[0]['azimuth']
    covtype2Azimuth = dataDict[1]['azimuth']
    covtype3Azimuth = dataDict[2]['azimuth']
    covtype4Azimuth = dataDict[3]['azimuth']
    covtype5Azimuth = dataDict[4]['azimuth']
    covtype6Azimuth = dataDict[5]['azimuth']
    covtype7Azimuth = dataDict[6]['azimuth']

    #creating simple if, elif, else that checks what covtype is forest with "azimuth" value and returns that covtype
    if azimuth < covtype6Azimuth:
        return 4
    elif azimuth < covtype3Azimuth:
        return 6
    elif azimuth < covtype5Azimuth:
        return 3
    elif azimuth < covtype1Azimuth:
        return 5
    elif azimuth < covtype2Azimuth:
        return 1
    elif azimuth < covtype7Azimuth:
        return 2
    else:
        return 7
    
dill.dump(heuristicPredict, open("heuristic.pickle", "ab"))


#Machine Learning


#creating function that trains model with "xTrain" and "yTrain" and then return score of model based on "xTest" and "yTest" and also return "yPredicted"
def fitScoreModelML(model, xTrain, yTrain, xTest, yTest):

    #fitting train dataset to model, getting score of model based on test dataset
    #and returning "score" and "yPredicted"
    model.fit(xTrain, yTrain.values.ravel())
    yPredicted = model.predict(xTest)
    score = model.score(xTest, yTest)
    return model, score, yPredicted

#getting scores of models and printing them
modelRFC, scoreRFC, yPredictedRFC = fitScoreModelML(RandomForestClassifier(), xTrainScaled, yTrain, xTestScaled, yTest)
modelKNN, scoreKNN, yPredictedKNN = fitScoreModelML(KNeighborsClassifier(), xTrainScaled, yTrain, xTestScaled, yTest)

#saving random forest classifier and k-nearest neighbors to pickle files
pickle.dump(modelRFC, open("modelRFC.pickle", "wb"))
pickle.dump(modelKNN, open("modelKNN.pickle", "wb"))


#Neural Network


#creating function that takes "epochs, learningrate, nodes" parameters and create and train model with them
def buildModelNN(epochs, learningRate, nodes):

    #this is function that stops training model when the validation loss isn't getting better
    earlyStopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min", patience=5,
                                        restore_best_weights=True)
    
    #creating model with 3 layers, first is input layer, second is hidden nodes layer and third is output layer
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(xTrainScaled.shape[1],)),
        keras.layers.Dense(nodes, activation="relu"),
        keras.layers.Dense(8, activation="softmax")
    ])

    #compiling, fitting and returning model and its history
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learningRate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(xTrainScaled, yTrain, epochs=epochs, validation_data=(xTestScaled, yTest), callbacks=[earlyStopping])
    return model, history

#this is code that hypertuned parameters for model and I commented it out,
#because it took a lot of time to compile, because my computer has too little 
#computing power
"""
scores = []

#for loop, which train models with different parameters
#taking epochs parameter between 5 and 8
for ep in range(5,9):
    #taking learning rate parameter with 0.001, 0.004, 0.007 values
    for lr in [x * 0.001 for x in range(1, 10, 3)]:
        #taking nodes parameter between 32 and 40
        for nd in range(32, 41):
        
            print("ep: " + str(ep) + ", lr: " + str(lr) + ", nd: " + str(nd))

            #building model with certain parameters by calling function buildModel()
            model, history = buildModelNN(ep, lr, nd)

            #saving loss and accuracy value of test data to "testLoss" and "testAcc" variables
            testLoss, testAcc = model.evaluate(xTestScaled, yTest)

            #appending parameters and their score to "scores" list
            scores.append({
                "epochs": ep,
                "learning_rate": lr,
                "nodes": nd,
                "score": testAcc
            })

            
#this is "scores" list after running for loop
[{'epochs': 5, 'learning_rate': 0.001, 'nodes': 32, 'score': 0.7203500270843506}, {'epochs': 5, 'learning_rate': 0.001, 'nodes': 33, 'score': 0.7233791947364807}, {'epochs': 5, 'learning_rate': 0.001, 'nodes': 34, 'score': 0.7235168814659119}, {'epochs': 5, 'learning_rate': 0.001, 'nodes': 35, 'score': 0.7197510600090027}, {'epochs': 5, 'learning_rate': 0.001, 'nodes': 36, 'score': 0.7253550887107849}, {'epochs': 5, 'learning_rate': 0.001, 'nodes': 37, 'score': 0.7206254005432129}, {'epochs': 5, 'learning_rate': 0.001, 'nodes': 38, 'score': 0.7301398515701294}, {'epochs': 5, 'learning_rate': 0.001, 'nodes': 39, 'score': 0.7215823531150818}, {'epochs': 5, 'learning_rate': 0.001, 'nodes': 40, 'score': 0.7226425409317017}, {'epochs': 5, 'learning_rate': 0.004, 'nodes': 32, 'score': 0.7197097539901733}, {'epochs': 5, 'learning_rate': 0.004, 'nodes': 33, 'score': 0.7267732620239258}, {'epochs': 5, 'learning_rate': 0.004, 'nodes': 34, 'score': 0.7268903255462646}, {'epochs': 5, 'learning_rate': 0.004, 'nodes': 35, 'score': 0.7204326391220093}, {'epochs': 5, 'learning_rate': 0.004, 'nodes': 36, 'score': 0.7171280384063721}, {'epochs': 5, 'learning_rate': 0.004, 'nodes': 37, 'score': 0.7254307866096497}, {'epochs': 5, 'learning_rate': 0.004, 'nodes': 38, 'score': 0.7333824634552002}, {'epochs': 5, 'learning_rate': 0.004, 'nodes': 39, 'score': 0.7249971032142639}, {'epochs': 5, 'learning_rate': 0.004, 'nodes': 40, 'score': 0.7353239059448242}, {'epochs': 5, 'learning_rate': 0.007, 'nodes': 32, 'score': 0.7249764204025269}, {'epochs': 5, 'learning_rate': 0.007, 'nodes': 33, 'score': 0.7199093699455261}, {'epochs': 5, 'learning_rate': 0.007, 'nodes': 34, 'score': 0.722456693649292}, {'epochs': 5, 'learning_rate': 0.007, 'nodes': 35, 'score': 0.7210316061973572}, {'epochs': 5, 'learning_rate': 0.007, 'nodes': 36, 'score': 0.720873236656189}, {'epochs': 5, 'learning_rate': 0.007, 'nodes': 37, 'score': 0.71966153383255}, {'epochs': 5, 'learning_rate': 0.007, 'nodes': 38, 'score': 0.7203706502914429}, {'epochs': 5, 'learning_rate': 0.007, 'nodes': 39, 'score': 0.7249488830566406}, {'epochs': 5, 'learning_rate': 0.007, 'nodes': 40, 'score': 0.7112830877304077}, {'epochs': 6, 'learning_rate': 0.001, 'nodes': 32, 'score': 0.7188835740089417}, {'epochs': 6, 'learning_rate': 0.001, 'nodes': 33, 'score': 0.7242466807365417}, {'epochs': 6, 'learning_rate': 0.001, 'nodes': 34, 'score': 0.7222157120704651}, {'epochs': 6, 'learning_rate': 0.001, 'nodes': 35, 'score': 0.7242397665977478}, {'epochs': 6, 'learning_rate': 0.001, 'nodes': 36, 'score': 0.7238610982894897}, {'epochs': 6, 'learning_rate': 0.001, 'nodes': 37, 'score': 0.7274479866027832}, {'epochs': 6, 'learning_rate': 0.001, 'nodes': 38, 'score': 0.7289006114006042}, {'epochs': 6, 'learning_rate': 0.001, 'nodes': 39, 'score': 0.728873074054718}, {'epochs': 6, 'learning_rate': 0.001, 'nodes': 40, 'score': 0.72984379529953}, {'epochs': 6, 'learning_rate': 0.004, 'nodes': 32, 'score': 0.7241984605789185}, {'epochs': 6, 'learning_rate': 0.004, 'nodes': 33, 'score': 0.7251347899436951}, {'epochs': 6, 'learning_rate': 0.004, 'nodes': 34, 'score': 0.7315098643302917}, {'epochs': 6, 'learning_rate': 0.004, 'nodes': 35, 'score': 0.7345321774482727}, {'epochs': 6, 'learning_rate': 0.004, 'nodes': 36, 'score': 0.7214102149009705}, {'epochs': 6, 'learning_rate': 0.004, 'nodes': 37, 'score': 0.7227045297622681}, {'epochs': 6, 'learning_rate': 0.004, 'nodes': 38, 'score': 0.72984379529953}, {'epochs': 6, 'learning_rate': 0.004, 'nodes': 39, 'score': 0.7334926128387451}, {'epochs': 6, 'learning_rate': 0.004, 'nodes': 40, 'score': 0.732019305229187}, {'epochs': 6, 'learning_rate': 0.007, 'nodes': 32, 'score': 0.7187183499336243}, {'epochs': 6, 'learning_rate': 0.007, 'nodes': 33, 'score': 0.7233448028564453}, {'epochs': 6, 'learning_rate': 0.007, 'nodes': 34, 'score': 0.7195926904678345}, {'epochs': 6, 'learning_rate': 0.007, 'nodes': 35, 'score': 0.7195858359336853}, {'epochs': 6, 'learning_rate': 0.007, 'nodes': 36, 'score': 0.7148423790931702}, {'epochs': 6, 'learning_rate': 0.007, 'nodes': 37, 'score': 0.7210659980773926}, {'epochs': 6, 'learning_rate': 0.007, 'nodes': 38, 'score': 0.7292517423629761}, {'epochs': 6, 'learning_rate': 0.007, 'nodes': 39, 'score': 0.7286527752876282}, {'epochs': 6, 'learning_rate': 0.007, 'nodes': 40, 'score': 0.7267526388168335}, {'epochs': 7, 'learning_rate': 0.001, 'nodes': 32, 'score': 0.7224016189575195}, {'epochs': 7, 'learning_rate': 0.001, 'nodes': 33, 'score': 0.724040150642395}, {'epochs': 7, 'learning_rate': 0.001, 'nodes': 34, 'score': 0.7255409359931946}, {'epochs': 7, 'learning_rate': 0.001, 'nodes': 35, 'score': 0.7300090193748474}, {'epochs': 7, 'learning_rate': 0.001, 'nodes': 36, 'score': 0.7269660234451294}, {'epochs': 7, 'learning_rate': 0.001, 'nodes': 37, 'score': 0.7290933728218079}, {'epochs': 7, 'learning_rate': 0.001, 'nodes': 38, 'score': 0.7286940813064575}, {'epochs': 7, 'learning_rate': 0.001, 'nodes': 39, 'score': 0.7318678498268127}, {'epochs': 7, 'learning_rate': 0.001, 'nodes': 40, 'score': 0.7293343544006348}, {'epochs': 7, 'learning_rate': 0.004, 'nodes': 32, 'score': 0.7268421053886414}, {'epochs': 7, 'learning_rate': 0.004, 'nodes': 33, 'score': 0.7242810726165771}, {'epochs': 7, 'learning_rate': 0.004, 'nodes': 34, 'score': 0.7286871671676636}, {'epochs': 7, 'learning_rate': 0.004, 'nodes': 35, 'score': 0.7251072525978088}, {'epochs': 7, 'learning_rate': 0.004, 'nodes': 36, 'score': 0.7276132106781006}, {'epochs': 7, 'learning_rate': 0.004, 'nodes': 37, 'score': 0.7200333476066589}, {'epochs': 7, 'learning_rate': 0.004, 'nodes': 38, 'score': 0.7342085838317871}, {'epochs': 7, 'learning_rate': 0.004, 'nodes': 39, 'score': 0.7261674404144287}, {'epochs': 7, 'learning_rate': 0.004, 'nodes': 40, 'score': 0.736005425453186}, {'epochs': 7, 'learning_rate': 0.007, 'nodes': 32, 'score': 0.7134379148483276}, {'epochs': 7, 'learning_rate': 0.007, 'nodes': 33, 'score': 0.7097960114479065}, {'epochs': 7, 'learning_rate': 0.007, 'nodes': 34, 'score': 0.7268971800804138}, {'epochs': 7, 'learning_rate': 0.007, 'nodes': 35, 'score': 0.7234480381011963}, {'epochs': 7, 'learning_rate': 0.007, 'nodes': 36, 'score': 0.7227802276611328}, {'epochs': 7, 'learning_rate': 0.007, 'nodes': 37, 'score': 0.7186701893806458}, {'epochs': 7, 'learning_rate': 0.007, 'nodes': 38, 'score': 0.7293412089347839}, {'epochs': 7, 'learning_rate': 0.007, 'nodes': 39, 'score': 0.7335752248764038}, {'epochs': 7, 'learning_rate': 0.007, 'nodes': 40, 'score': 0.7315855622291565}, {'epochs': 8, 'learning_rate': 0.001, 'nodes': 32, 'score': 0.7221331000328064}, {'epochs': 8, 'learning_rate': 0.001, 'nodes': 33, 'score': 0.7269384860992432}, {'epochs': 8, 'learning_rate': 0.001, 'nodes': 34, 'score': 0.718690812587738}, {'epochs': 8, 'learning_rate': 0.001, 'nodes': 35, 'score': 0.7317920923233032}, {'epochs': 8, 'learning_rate': 0.001, 'nodes': 36, 'score': 0.7308558225631714}, {'epochs': 8, 'learning_rate': 0.001, 'nodes': 37, 'score': 0.7371000647544861}, {'epochs': 8, 'learning_rate': 0.001, 'nodes': 38, 'score': 0.7278197407722473}, {'epochs': 8, 'learning_rate': 0.001, 'nodes': 39, 'score': 0.7305116057395935}, {'epochs': 8, 'learning_rate': 0.001, 'nodes': 40, 'score': 0.7277990579605103}, {'epochs': 8, 'learning_rate': 0.004, 'nodes': 32, 'score': 0.7207080125808716}, {'epochs': 8, 'learning_rate': 0.004, 'nodes': 33, 'score': 0.7254652380943298}, {'epochs': 8, 'learning_rate': 0.004, 'nodes': 34, 'score': 0.7194068431854248}, {'epochs': 8, 'learning_rate': 0.004, 'nodes': 35, 'score': 0.7229041457176208}, {'epochs': 8, 'learning_rate': 0.004, 'nodes': 36, 'score': 0.7232621908187866}, {'epochs': 8, 'learning_rate': 0.004, 'nodes': 37, 'score': 0.7344771027565002}, {'epochs': 8, 'learning_rate': 0.004, 'nodes': 38, 'score': 0.7294582724571228}, {'epochs': 8, 'learning_rate': 0.004, 'nodes': 39, 'score': 0.7390621900558472}, {'epochs': 8, 'learning_rate': 0.004, 'nodes': 40, 'score': 0.7317783236503601}, {'epochs': 8, 'learning_rate': 0.007, 'nodes': 32, 'score': 0.71907639503479}, {'epochs': 8, 'learning_rate': 0.007, 'nodes': 33, 'score': 0.7215685844421387}, {'epochs': 8, 'learning_rate': 0.007, 'nodes': 34, 'score': 0.7276545166969299}, {'epochs': 8, 'learning_rate': 0.007, 'nodes': 35, 'score': 0.7226219177246094}, {'epochs': 8, 'learning_rate': 0.007, 'nodes': 36, 'score': 0.7209765315055847}, {'epochs': 8, 'learning_rate': 0.007, 'nodes': 37, 'score': 0.7225255370140076}, {'epochs': 8, 'learning_rate': 0.007, 'nodes': 38, 'score': 0.7342292666435242}, {'epochs': 8, 'learning_rate': 0.007, 'nodes': 39, 'score': 0.7333893179893494}, {'epochs': 8, 'learning_rate': 0.007, 'nodes': 40, 'score': 0.7330175638198853}]

#for loop to get highest score with its parameters
for i in scores:
    max = []
    if max:
        if max[0]["score"] < i["score"]:
            max[0] = i
    else:
        max.append(i)

#this is highest score with its parameters and those parameters I will use
[{'epochs': 8,
  'learning_rate': 0.007,
  'nodes': 40,
  'score': 0.7330175638198853}] 

"""

#creating neural network model and its history by using best parameters
modelNN, historyNN = buildModelNN(8, 0.007, 40)

#saving neural network model
modelNN.save("modelNN")

#creating plot which shows training curves for neural network model
plt.plot(historyNN.history['accuracy'])
plt.plot(historyNN.history['val_accuracy'])
plt.title('Neural Network Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


#Evaluating Models


#getting predicted classes by neural network model
yProbs = modelNN.predict(xTestScaled)
yPredictedNN = yProbs.argmax(axis=-1)

#getting loss and score of neural network model
lossNN, scoreNN = modelNN.evaluate(xTestScaled, yTest)

#printing scores of each model
print("Score of Random Forest Classifier: " + str(scoreRFC) + ", Score of K-Nearest Neighbors: " + str(scoreKNN) + ", Score of Neural Network: " + str(scoreNN))

#creating confusion matrixes of each model 
cmRFC = confusion_matrix(yTest, yPredictedRFC)
cmKNN = confusion_matrix(yTest, yPredictedKNN)
cmNN = confusion_matrix(yTest, yPredictedNN)

#creating confusion matrix heatmap for random forest classifier
plt.figure(figsize=(10, 7))
sn.heatmap(cmRFC, annot=True)
plt.title("Confusion Matrix of Random Forest Classifier")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()

#creating confusion matrix heatmap for k-nearest neighbors
plt.figure(figsize=(10, 7))
sn.heatmap(cmKNN, annot=True)
plt.title("Confusion Matrix of K-Nearest Neighbors")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()

#creating confusion matrix heatmap for neural network
plt.figure(figsize=(10, 7))
sn.heatmap(cmNN, annot=True)
plt.title("Confusion Matrix of Neural Network")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()



