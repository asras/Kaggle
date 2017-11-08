import pandas as pd
import numpy as np
import tensorflow as tf
from NN import NN


def fill_avg_age_for_nans(ldf):
	average_age = ldf["Age"].mean()
	ldf["Age"] = [average_age if bl else ldf["Age"].values[j] for j, bl in enumerate(ldf["Age"].isnull())]
	return ldf


def map_from_port_to_num(port):
    if port == "C":
        return 0
    elif port == "Q":
        return 1
    elif port == "S":
        return 2
    else:
        print("Incorrect value: {}".format(port))
        raise ValueError

def fill_most_common_port_for_nans(ldf):
	ldf = ldf.dropna()
	most_common_port = ldf["Embarked"].value_counts().idxmax()	
	ldf["Embarked"] = [most_common_port if bl else ldf["Embarked"].values[j] \
	                            for j, bl in enumerate(ldf["Embarked"].isnull())]
	ldf["Embarked"] = [map_from_port_to_num(p) for p in ldf["Embarked"].values]
	return ldf

def female_to_zero_male_to_one(ldf):
	ldf["Sex"] = [0 if s == "female" else 1 for s in ldf["Sex"]]
	return ldf



df = pd.read_csv("test.csv")
df_answer = df["PassengerId"]
df.drop(["Name", "PassengerId", "Cabin", "Ticket"], 1, inplace=True)

df = fill_avg_age_for_nans(df)
df = fill_most_common_port_for_nans(df)
df = female_to_zero_male_to_one(df)

X_array = np.array([row.reshape([7]) for row in df.values])



##
sess = tf.Session()
aNN = NN(sess)





preds = aNN.predict(sess, X_array)

df_answer["Survived"] = preds

df_answer.to_csv("submission.csv", header=True, index = False)