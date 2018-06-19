import pandas as pd
import numpy as np
import string
import random


#Before working on pandas DataFrame we must understand how it works.
#The DataFrame's basic structure is the dictionary {"key" ; value}, we may need to create a dataset from a DataFrame
# We can do it creating a dictionary and wrapping it on a pandas DataFrame
# The next two function allow you to do that.

#This function allow you to create a basic dictionary, it has some key (A,B,C,D and testname) and some random values
def create_dict():

    dictionary = {"A": np.random.randint(1, 10),
            "B": np.random.randint(1, 10) ,
            "C": np.random.randint(1, 10)** -np.random.randint(10) ,
            "D": np.random.randint(1, 10),
            "testname" :  "test."+random.choice(string.ascii_letters)}

    return dictionary


#In this test case we use a for to create our data in an easy an quick manner.
# We can se how the dict is wrapped on a DataFrame as described above
def create_DataFrame(n_dataframe = 100):

    population = []
    for i in range(n_dataframe):
        population.append(pd.DataFrame(create_dict(), index=[i]))

    return population

# We need to call the functions
dataset_test = create_DataFrame(100)

# The output of create_DataFrame is a long list of DataFrame, we have to concatenate the DataFrames in one DataFrame
dataset_test  = pd.concat(dataset_test )
#As result we have our Pandas DataFrame and we may need to stored in a file.
dataset_test .to_csv("dataset_test.csv", sep=';', header=True, float_format='%.6f',index=False)


# We need to provide a file path where the csv is stored. '/home/dataset.csv'

# The function read_csv from pandas allow us to open a csv. This function will accept in our case three parameters described as following

# arg1-> Path -> path_file indicates where is placed
# arg2-> Sep  -> indicates where a new data start and finish in other words how the data is divided.
# arg3-> header -> refers where the dataset features is placed (horizontally, columns-wise)

#If we already have a dataset stored in CSV format we cat load it as following.

dataset_org = pd.read_csv("dataset_test.csv", sep=";", header=0)

#Pandas allow us to display the result in many formats, in this case 4f->(2.0121)
pd.options.display.float_format = '{:.4f}'.format

#In order to manipulate our dataset in a intuitive manner we will wrap the csv file in a pandas DataFrame
dataset = pd.DataFrame(dataset_org)


#So far, we have loaded the dataset and wrapped it on a DataFrame using pandas' function set.
#Our dataset is now a DataFrame then we can use many function from pandas.

#If we need to take at look of what we have loaded we can see it doing :

#print the hole dataset
print (dataset)
# #print the fist n elements
print (dataset.head(10))
# # print the last n elements
print (dataset.tail(10))
# # We can also have a statistical description of the first n elements doing
print(dataset.head(10).describe())
# # The same information but taking the whole dataset
print(dataset.describe())


# We want to take all the elements whose come from test.d thus wi will apply a filter.
# Dataset.Feature.str.contains('key')
# We want to store only the elements that contain the same "key" in our case the letter "d"
dataset= dataset[dataset.testname.str.contains("test.d") == True]



dataset= dataset[dataset.A>=0]

dataset= dataset[dataset.B<=8]

#Print to check if the filers have worked well
print(dataset.B.describe())

# We use the function to_csv to write our DataFrame in a file called filtered_dataset
#As we have mentioned earlier we will need to specify sep, header position (feature position),
# float precision, and index if we need

df = dataset[["A","B"]].copy()
print(df)
df.to_csv("filtered_dataset.csv", sep=';', header=True, float_format='%.4f',index=False)


#importing a graphic library called pyplot from matplotlib under the pseudonym plt
from matplotlib import pyplot as plt

#To show a pretty chart we need to declare some items like title, ylabels, etc.. as described below
plt.title(' Vdd Vs Column 23')
plt.ylabel("Column 23")
plt.xlabel("Vdd ")

# We need to declare 'X' and 'Y', in our case X = Vdd and Y = C23 thus we have to take the relevant elements from the DataFrame
A = dataset.A
B = dataset.B

# We may need to divide our axis in order to  have a more intuitive view
# We should create two array that will store the resolution of our axis, from max() to min() and the period (in this case 10)(optional)
print(A)
major_ticks = np.arange(min(A), max(A), 1)
minor_ticks = np.arange(min(B), max(B), 1)
# We must load the arrays in xticks
plt.xticks(minor_ticks)
plt.yticks(major_ticks)
#
#We may need to have a grid to locate our values on the picture, we can declare this grid as following: (optional)

plt.grid(which='both')
plt.grid(which='minor', alpha=0.2, color='r', linestyle='-.', linewidth=0.5) #alpha = 0.0 transparent through 1.0 opaque

plt.grid(which='major', alpha=0.2, color='r', linestyle='-.', linewidth=0.5) # linewidth = thickness


#We load the values (X and Y),
# arg1 = VDD -> X
# arg2 =  C23 -> Y
# arg3 =  linestyle + color
# arg5 = label name's line
plt.plot(A, B, ('o' + 'r'), linewidth=0.5, label="AB" )

#plot legand
plt.legend()
# sowh the picutre
plt.show()



