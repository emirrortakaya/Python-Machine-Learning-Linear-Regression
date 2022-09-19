import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.metrics as mt

data = pd.read_csv("linear-regression/linear-regression-dataset.csv")

deneyim = data["deneyim"].values.reshape(-1,1)
maas = data["maas"].values.reshape(-1,1)
#algoritma
reg = lm.LinearRegression()
#data_split
x_train,x_test,y_train,y_test = ms.train_test_split(deneyim,maas,test_size=1/3, random_state=0)
#train
reg.fit(x_train, y_train)
#predict
y_pred = reg.predict(x_test)

print("deneyimler:", x_test )
print("Tahmin Edilen:", y_pred)

#score
score = mt.r2_score(y_test,y_pred)
print("Skor:",score)

#graph
plt.scatter(deneyim,maas,color="r")
plt.plot(x_test,y_pred, color="b")
plt.show()