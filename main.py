# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


#%%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
#%%
data = {'marks': [55,21,63,88,74,54,95,41,84,52],
        'grade': ['average','poor','average','good','good','average','good','average','good','average'],
        'point': ['c','f','c+','b+','b','c','a','d+','b+','c']}
df = pd.DataFrame(data)

#%%

def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(data).numpy())
#%%
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
