import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pyod.models.knn import KNN

# define the dataset location
filename = 'glass+identification/glass.data'
# load the csv file as a data frame
df = pd.read_csv(filename, header=None)
df.columns = ["Id_Number", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type_of_glass"]
df = df.drop('Id_Number', axis=1)

# boxplot
# for column in df:
#     plt.figure()
#     df.boxplot([column])
#     plt.show()

# # clustering based technique
# km = KMeans(n_clusters=4)
# # reshape
# for column in df:
#     hdata = df[column].values.reshape(-1, 1)
#     km.fit(hdata)
#
#     print(column)
#     print(km.labels_)
#
#     plt.scatter(df[column], range(0, 214), c=km.labels_)
#     plt.title(column)
#     plt.show()

# KNN approach
allOutScore = pd.DataFrame()
for column in df:
    hdata = df[column].values.reshape(-1, 1)

    # pyod library KNN defined
    mdl = KNN(n_neighbors=2, contamination=0.02)

    mdl.fit(hdata)

    # print(column)
    # print(mdl.labels_)

    plt.scatter(df[column], range(0, 214), c=mdl.labels_)
    plt.title(column)
    plt.show()

    # examine the outlier score
    # print(column)
    # print(mdl.decision_scores_)

    outScore = mdl.decision_scores_

    allOutScore[column] = outScore

# allOutScore.to_csv('out.csv', index=False)

allOutScore[(allOutScore > 1).any(1)]