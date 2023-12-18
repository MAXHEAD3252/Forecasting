#Importing necessary libraries
from apyori import apriori
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

#reading dataset from CSV file
store_data =  pd.read_csv('F:\Learning_Work\Vs_Work\DM_Project\dataset.csv',header=None)
print(store_data)

#displaying the shape of the dataset
print(store_data.shape)

#Initializing an empty list called 'records'
records = []
#Looping through the dataset and creating a list of items for each transaction
for i in range(0,5):
    records.append([str(store_data.values[i,j])  for j in range(0,4)])
    
#Applying the Apriori algorithm on the records with the specified minimum support and minimum confidence
association_rule = apriori(records,min_support=0.2,min_confidence = 0.6)
#converting the resulting associatio rules into a list
association_results = list(association_rule)

#printing the number of association rules
print(len(association_results))
#Printing the association rules
print('\n'.join(map(str,association_results)))

# create the graph
G = nx.DiGraph()

# add nodes to the graph
for result in association_results:
    for item in result[0]:
        G.add_node(item)

# add edges to the graph
for result in association_results:
    for item in result[0]:
        for item2 in result[0]:
            if item != item2:
                G.add_edge(item, item2)

# draw the graph
nx.draw(G, with_labels=True)
plt.show()