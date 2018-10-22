# Project: Identify Customer Segments
 
In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
 
```python
# import libraries here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')
# Set base plotting style
plt.style.use('seaborn-ticks')
# Set base plotting size
plt.rcParams['figure.figsize'] = 12, 9
# Increase figure resolution for high dpi screens
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
```

### Step 0: Load the Data

There are four files associated with this project (not including this one):
 
- `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
- `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
- `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
- `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns

Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.

To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.

Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

```python
# Load in the general demographics data.
azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv', delimiter=";")

# Load in the feature summary file.
feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', delimiter=";")

# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).

print(azdias.shape,"\n")
print(azdias.columns,"\n")
print(azdias.info(),"\n")
print(azdias.describe(), "\n")

print(feat_info.shape,"\n")
print(feat_info.columns,"\n")
print(feat_info.info(),"\n")
print(feat_info.describe(), "\n")
```

```
(85, 4) 

Index(['attribute', 'information_level', 'type', 'missing_or_unknown'], dtype='object') 

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 85 entries, 0 to 84
Data columns (total 4 columns):
attribute             85 non-null object
information_level     85 non-null object
type                  85 non-null object
missing_or_unknown    85 non-null object
dtypes: object(4)
memory usage: 2.7+ KB
None 

            attribute information_level     type missing_or_unknown
count              85                85       85                 85
unique             85                 9        5                  9
top     FINANZ_SPARER            person  ordinal               [-1]
freq                1                43       49                 26
```

> **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 

## Step 1: Preprocessing

### Step 1.1: Assess Missing Data

The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!

#### Step 1.1.1: Convert Missing Value Codes to NaNs
The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.

**As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

```python
feat_info.head(10)
```
|   | attribute | information_level | type | missing_or_unknown |
| --- | --- | --- | --- | --- |
| 0 | AGER_TYP | person | categorical | [-1,0] |
| 1 | ALTERSKATEGORIE_GROB | person | ordinal | [-1,0,9] |

```python
# Not sure what to do with the X in object variables yet
print(azdias['CAMEO_DEUG_2015'].value_counts(),"\n")

att = np.array(feat_info['attribute'])
miss = np.array(feat_info['missing_or_unknown'])
```
```
8    134441
9    108177
6    105874
4    103912
3     86779
2     83231
7     77933
5     55310
1     36212
X       373
Name: CAMEO_DEUG_2015, dtype: int64 
```
```python
### https://thispointer.com/python-how-to-replace-single-or-multiple-characters-in-a-string/
# Replace a set of multiple sub strings with a new string in main string.
def replaceMultiple(mainString, toBeReplaces, newString):
    # Iterate over the strings to be replaced
    for elem in toBeReplaces :
        # Check if string is in the main string
        if elem in mainString :
            # Replace the string
            mainString = mainString.replace(elem, newString)
    
    return  mainString

def f_number(s):
    try:
        int(s)
        return int(s)
    except ValueError:
        return s
```

```
miss1 = [replaceMultiple(s, ['[', ']'] , "") for s in miss]
miss2 = [s.split(',') for s in miss1]
xmiss = [[f_number(y) for y in x] for x in miss2]

print("Number of naturally missing observations: {:,}".format(azdias.isnull().sum().sum()))
natmiss = azdias.isnull().sum().sum()
```
```
Number of naturally missing observations: 4,896,838
```
```python
# Identify missing or unknown data values and convert them to NaNs.
n = np.NaN

# loop through the 'att' and 'miss' arrays to convert the missing values to NaNs
for i in range(len(xmiss)):
    azdias[att[i]] = azdias[att[i]].replace(xmiss[i],n)

print("Number of artificially missing observations: {:,}".format(azdias.isnull().sum().sum() - natmiss))
print("Total number of missing observations: {:,}".format(azdias.isnull().sum().sum()))
```
```
Number of artificially missing observations: 3,477,091
Total number of missing observations: 8,373,929
```
#### Step 1.1.2: Assess Missing Data in Each Column

How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)

For the remaining features, are there any patterns in which columns have, or share, missing data?

```python
# Perform an assessment of how much missing data there is in each column of the
# dataset.
col_nulls = azdias.isnull().sum() / len(azdias)

# the histogram of the data
n, bins, patches = plt.hist(col_nulls,bins=100)

plt.xlabel('Number of missing values')
plt.ylabel('Frequency of variables')
plt.title('Distribution of missing values')

def hist_me(xy):
    plt.annotate('Craziness!!', xy=(xy, 1), xytext=(0.85, 21),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

hist_me(0.997576)
hist_me(0.769554)
hist_me(0.655967)
hist_me(0.534687)
hist_me(0.440203)
hist_me(0.348137)

plt.grid(True)
plt.show()
```
<p align="center"><img src="/img/1.png?raw=true"/></p>

```python
# Investigate patterns in the amount of missing data in each column.

col_nulls = col_nulls.sort_values(ascending=False)
print(col_nulls.head(6))
names = list(col_nulls.index)
print(names[:6])

# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)

cols_drop = names[:6]
azdias = azdias.drop(cols_drop, axis=1, inplace=False)
print(azdias.shape)
```
```
(891221, 79)
```

#### Step 1.1.3: Assess Missing Data in Each Row

Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.

In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.

- You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
- To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.

Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

```python
# How much data is missing in each row of the dataset?

azdias['NaN_count'] = azdias.isnull().sum(axis=1)
azdias['NaN_count'][:5]

n, bins, patches = plt.hist(azdias['NaN_count'])
plt.xlabel('Number of missing values')
plt.ylabel('Frequency of variables')
plt.title('Distribution of missing values')
plt.show()
```
<p align="center"><img src="/img/2.png?raw=true"/></p>

```python
col_n = col_nulls.sort_values(ascending=True)
print(col_n.head(30))
```
`print thing`
```python
# Write code to divide the data into two subsets based on the number of missing
# values in each row.

azdias_highNaN = azdias.loc[azdias.NaN_count>30]
azdias_lowNaN = azdias.loc[azdias.NaN_count<=30]
print(azdias_highNaN.shape, azdias_lowNaN.shape)
```
```
(93154, 80) (798067, 80)
```

```python
def plot_sns(desc,var):
    for d , v in zip(desc, var):
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        sns.countplot(data=azdias_highNaN, x = v, ax=ax1)
        sns.countplot(data=azdias_lowNaN, x = v, ax=ax2)
        ax1.set_title("High number of NaNs")
        ax2.set_title("Low number of NaNs")
        ax2.label_outer()

plot_sns(["Estimated age based on given name analysis",
         "Energy consumption typology",
         "Most descriptive financial type for individual",
         "Personality typology",
         "Gender"],
         ["ALTERSKATEGORIE_GROB","ZABEOTYP","FINANZTYP","SEMIO_LUST","ANREDE_KZ"])
```
<p align="center"><img src="/img/3.png?raw=true"/></p>
<p align="center"><img src="/img/4.png?raw=true"/></p>
<p align="center"><img src="/img/5.png?raw=true"/></p>
<p align="center"><img src="/img/6.png?raw=true"/></p>
<p align="center"><img src="/img/7.png?raw=true"/></p>

### Step 1.2: Select and Re-Encode Features

Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.

- For numeric and interval data, these features can be kept without changes.
- Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
- Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.

In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.

Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

```python
# How many features are there of each data type?
feat_info["type"].value_counts()
```
```
ordinal        49
categorical    21
mixed           7
numeric         7
interval        1
Name: type, dtype: int64
```

#### Step 1.2.1: Re-Encode Categorical Features

For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:

- For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
- There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
- For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

```python
# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?

# drop the same features with the absurd number of NaNs from feat_info
feat_info = feat_info[~feat_info['attribute'].isin(names[:6])]
feat_infoc = feat_info[(feat_info["type"] == "categorical")]
print([azdias_lowNaN[s].value_counts() for s in feat_infoc['attribute']])

arr = np.array(feat_infoc['attribute'])
print(arr)

arr = np.delete(arr, np.where(arr=='OST_WEST_KZ'))
```
`print thing`
```python
# Re-encode categorical variable(s) to be kept in the analysis.
azdias_lowNaN = azdias_lowNaN.drop(arr, axis=1, inplace=False)

binary_nums = {"OST_WEST_KZ": {"O": 0, "W": 1}}
azdias_lowNaN = azdias_lowNaN.replace(binary_nums, inplace=False)
azdias_lowNaN["OST_WEST_KZ"].head()
```
`print thing`

#### Step 1.2.2: Engineer Mixed-Type Features

There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:

- "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
- "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
- If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.

Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

```python
print(azdias_lowNaN["PRAEGENDE_JUGENDJAHRE"].head(), "\n")

# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
from math import isnan
feat_infom = feat_info[(feat_info["type"] == "mixed")]
arrr = np.array(feat_infom['attribute'])
arrr = np.delete(arrr, np.where(arrr=='PRAEGENDE_JUGENDJAHRE'))
arrr = np.delete(arrr, np.where(arrr=='CAMEO_INTL_2015'))
azdias_lowNaN = azdias_lowNaN.drop(arrr, axis=1, inplace=False)

feat_infom['attribute']
```

```python
decade_nums = {1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:2, 8:3, 9:3, 10:4, 11:4, 12:4, 13:4, 14:5, 15:5}
move_nums = {1:1, 2:0, 3:1, 4:0, 5:1, 6:0, 7:0, 8:1, 9:0, 10:1, 11:0, 12:1, 13:0, 14:1, 15:0}

print(col_nulls["PRAEGENDE_JUGENDJAHRE"])
azdias_lowNaN["DECADE"] = [s if isnan(s) else decade_nums[s] for s in azdias_lowNaN['PRAEGENDE_JUGENDJAHRE']]
azdias_lowNaN["MOVEMENT"] = [s if isnan(s) else move_nums[s] for s in azdias_lowNaN['PRAEGENDE_JUGENDJAHRE']]
azdias_lowNaN[["DECADE","MOVEMENT"]].head()
```

```python
def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

# Investigate "CAMEO_INTL_2015" and engineer two new variables.
# Variables left as strings
azdias_lowNaN["HOUSE_WEALTH"] = [int(str(s)[0]) if is_number(s) else s for s in azdias_lowNaN['CAMEO_INTL_2015']]
azdias_lowNaN["FAM_SITUATION"] = [int(str(s)[1]) if is_number(s) else s for s in azdias_lowNaN['CAMEO_INTL_2015']]
azdias_lowNaN[["HOUSE_WEALTH","FAM_SITUATION"]].head()

feat_info[(feat_info["type"] == "mixed")]
```
|   | attribute | information_level | type | missing_or_unknown |
| --- | --- | --- |--- | --- |
|15	| LP_LEBENSPHASE_FEIN |	person | mixed | [0] |
|16	| LP_LEBENSPHASE_GROB | person | mixed | [0] |
|22	| PRAEGENDE_JUGENDJAHRE	| person | mixed | [-1,0] |
|56	| WOHNLAGE | building | mixed | [-1] |
|59	| CAMEO_INTL_2015 | microcell_rr4 | mixed | [-1,XX] |
|79 | PLZ8_BAUMAX | macrocell_plz8	| mixed | [-1,0] |

#### Step 1.2.3: Complete Feature Selection

In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:

- All numeric, interval, and ordinal type columns from the original dataset.
- Binary categorical features (all numerically-encoded).
- Engineered features from other multi-level categorical features and mixed features.

Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

```python
# Do whatever you need to in order to ensure that the dataframe only contains
# the columns that should be passed to the algorithm functions.

azdias_lowNaN = azdias_lowNaN.drop("PRAEGENDE_JUGENDJAHRE", axis=1, inplace=False)
azdias_lowNaN = azdias_lowNaN.drop("CAMEO_INTL_2015", axis=1, inplace=False)
azdias_lowNaN = azdias_lowNaN.drop("NaN_count", axis=1, inplace=False)
```

### Step 1.3: Create a Cleaning Function

Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

```python
# Helper functions

def replaceMultiple(mainString, toBeReplaces, newString):
    # Iterate over the strings to be replaced
    for elem in toBeReplaces :
        # Check if string is in the main string
        if elem in mainString :
            # Replace the string
            mainString = mainString.replace(elem, newString)
    
    return  mainString

def re_encode(df):
    binary_nums = {"OST_WEST_KZ": {"O": 0, "W": 1}}
    df.replace(binary_nums, inplace=True)
    df["DECADE"] = [s if isnan(s) else decade_nums[s] for s in df['PRAEGENDE_JUGENDJAHRE']]
    df["MOVEMENT"] = [s if isnan(s) else move_nums[s] for s in df['PRAEGENDE_JUGENDJAHRE']]
    df["HOUSE_WEALTH"] = [int(str(s)[0]) if is_number(s) else s for s in df['CAMEO_INTL_2015']]
    df["FAM_SITUATION"] = [int(str(s)[1]) if is_number(s) else s for s in df['CAMEO_INTL_2015']]
    return df
```
```python
def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', delimiter=";")
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    n = np.NaN
    att = np.array(feat_info['attribute'])
    m = np.array(feat_info['missing_or_unknown'])
    m1 = [replaceMultiple(s, ['[', ']'] , "") for s in m]
    m2 = [s.split(',') for s in m1]
    xm = [[f_number(y) for y in x] for x in m2]
    for i in range(len(xm)):
        df[att[i]] = df[att[i]].replace(xm[i],n)
    
    # select, re-encode, and engineer column values.
    df = re_encode(df)
    
    # Return the cleaned dataframe.
    df = df.drop(names[:6], axis=1, inplace=False)
    df['NaN_count'] = df.isnull().sum(axis=1)
    df_highNaN = df.loc[df.NaN_count>30]
    df = df.loc[df.NaN_count<=30]
    df = df.drop("NaN_count", axis=1, inplace=False)
    
    feat_info = feat_info[~feat_info['attribute'].isin(names[:6])]
    feat_infoc = feat_info[(feat_info["type"] == "categorical")]
    arr = np.array(feat_infoc['attribute'])
    arr = np.delete(arr, np.where(arr=='OST_WEST_KZ'))
    df = df.drop(arr, axis=1, inplace=False)
    
    feat_infom = feat_info[(feat_info["type"] == "mixed")]
    arrr = np.array(feat_infom['attribute'])
    df = df.drop(arrr, axis=1, inplace=False)
    return df, df_highNaN
```

## Step 2: Feature Transformation

### Step 2.1: Apply Feature Scaling

Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:

- sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
- For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
- For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

```python
# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.

from sklearn import preprocessing as p
print(azdias_lowNaN.isnull().sum().head())

# Still need to deal with the categorical variables first so all I have are numbers!
# Find a way to temporarily remove missing values in order to compute the scaling 
# parameters before re-introducing those missing values and applying imputation
from sklearn.preprocessing import Imputer
azdiasFit = Imputer(missing_values="NaN",strategy="mean",axis=0).fit(azdias_lowNaN)
azdiasTT = azdiasFit.transform(azdias_lowNaN)
cols = azdias_lowNaN.columns

# Apply feature scaling to the general population demographics data.

fit_az = p.StandardScaler().fit(azdiasTT) # Fit the data
trans_az = fit_az.transform(azdiasTT) #  Transform the data
df_az = pd.DataFrame(trans_az) # create a dataframe
df_az.columns = cols # add column names again
print(df_az.head())
```
```
ALTERSKATEGORIE_GROB    2803
FINANZ_MINIMALIST          0
FINANZ_SPARER              0
FINANZ_VORSORGER           0
FINANZ_ANLEGER             0
dtype: int64
```

### Step 2.2: Perform Dimensionality Reduction

On your scaled data, you are now ready to apply dimensionality reduction techniques.

- Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
- Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
- Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

```python
# Apply PCA to the data.

from sklearn.decomposition import PCA

pca = PCA()
Xall_pca = pca.fit(df_az)
```
```python
# Taken from the lecture files
def do_pca(n_components, data):
    '''
    Transforms data using PCA to create n_components, and provides back the results of the
    transformation.

    INPUT: n_components - int - the number of principal components to create
           data - the data you would like to transform

    OUTPUT: pca - the pca object created after fitting the data
            X_pca - the transformed X matrix with new number of components
    '''
    pca = PCA(n_components)
    X_pca = pca.fit_transform(data)
    return pca, X_pca

# Taken from the lecture files
def pca_results(full_dataset, pca):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''

    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = full_dataset.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)
```

```python
# Investigate the variance accounted for by each principal component.

# Dimension indexing
dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
# PCA explained variance
ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
variance_ratios.index = dimensions
variance_ratios[:5]
```
|   | Explained Variance |
| --- | --- |
| Dimension 1 | 0.1773 |
| Dimension 2 | 0.1424 |
| Dimension 3 | 0.0871 |
| Dimension 4 | 0.0484 |
| Dimension 5 | 0.0399 |

```python
targ = 0.80
for comp in range(3, df_az.shape[1]):
    pca, Xall_pca = do_pca(comp, df_az)
    comp_check = pca_results(df_az, pca)
    if comp_check['Explained Variance'].sum() > targ:
        break

actual_pct = comp_check['Explained Variance'].sum()
num_comps = comp_check.shape[0]
print("Using {0:} components, we can explain {1:3.2f}% of the variability in the original data.".format(num_comps,actual_pct))
```
```
Using 21 components, we can explain 0.8099000000000002% of the variability in the original data.
```
```python
from matplotlib.pyplot import plot

c_ratios = np.cumsum(ratios)
fig = plot(c_ratios, "b")
plt.axvline(x=num_comps-1, color="r", linestyle="--")
plt.axhline(y=actual_pct, color="r", linestyle="--")
plt.title("Cumulative variance explained by the components")
plt.xlabel("Number of components")
plt.ylabel("Cumulative percentage")
plt.annotate('Perfection!', xy=(num_comps-1, actual_pct), xytext=(25, 0.7),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )
plt.show()
```
<p align="center"><img src="/img/8.png?raw=true"/></p>

```python
# Re-apply PCA to the data while selecting for number of components to retain.

pca = PCA(num_comps)
pcafit = pca.fit(df_az) # for later use
X_pca = pcafit.transform(df_az)
```

### Step 2.3: Interpret Principal Components

Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.

As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.

- To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
- You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

```python
# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.

def top_bott(component_number, data):
    comps = pca.components_[component_number - 1].reshape(len(pca.components_[component_number - 1]), 1)
    df_cmp = pd.DataFrame(np.round(comps, 4), columns = ["feature_weights"])
    df_cmp.index = data.columns
    df_cmp.sort_values("feature_weights",inplace=True, ascending=False)
    print("Top 5 positive weights for component {}\n".format(component_number), df_cmp.head(5), "\n")
    print("Top 5 negative weights for component {}\n".format(component_number), df_cmp.tail(5), "\n")
    
top_bott(1, df_az)

# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.

top_bott(2, df_az)

# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.

top_bott(3, df_az)

top_bott(0, df_az)
top_bott(8, df_az)
top_bott(11, df_az)
```
Top 5 positive weights for component 1

|   | feature_weights |
| --- | --- |
| PLZ8_ANTG3 | 0.2300 |
| PLZ8_ANTG4 | 0.2230 |
| HOUSE_WEALTH | 0.2072 |
| ORTSGR_KLS9 | 0.2028 |
| EWDICHTE | 0.2010  |

Top 5 negative weights for component 1

|   | feature_weights |
| --- | --- |
| KBA05_GBZ | -0.2182 |
| FINANZ_MINIMALIST | -0.2182 |
| KBA05_ANTG1 | -0.2260 |
| PLZ8_ANTG1 | -0.2289 |
| MOBI_REGIO | -0.2431 |

## Step 3: Clustering

### Step 3.1: Apply Clustering to General Population

You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.

- Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
- Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
- Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
- Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

```python
# Over a number of different cluster counts...
    # run k-means clustering on the data and...
    # compute the average within-cluster distances.
    
from sklearn.cluster import KMeans
def fit_kmeans(data, centers):
    '''
    INPUT:
        data = the dataset you would like to fit kmeans to (dataframe)
        centers = the number of centroids (int)
    OUTPUT:
        labels - the labels for each datapoint to which group it belongs (nparray)
    
    '''
    dicts = {}
    for i in centers:
        kmeans = KMeans(i)
        labels = kmeans.fit(data)
        dicts[i] = labels.inertia_
    return dicts

d = fit_kmeans(X_pca, range(10, num_comps, 1))
print(d)
```

```python
# Investigate the change in within-cluster distance across number of clusters.
# HINT: Use matplotlib's plot function to visualize this relationship.

lists = sorted(d.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)
plt.axvline(x=12, color="k", linestyle="--")
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster distance')
plt.title('Population Clusters')
plt.show()
```
<p align="center"><img src="/img/9.png?raw=true"/></p>

```python
# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.
kk=12
kmeans = KMeans(kk)
kfit = kmeans.fit(X_pca)
X_kmeans = kfit.predict(X_pca)
```

### Step 3.2: Apply All Steps to the Customer Data

Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.

- Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
- Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
- Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

```python
# Load in the customer demographics data.
customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv', delimiter=";")

# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.

df_cus, cus_highNaN = clean_data(customers)

# imputes missing values
cus_imp = azdiasFit.transform(df_cus)

# standardizes values
cus_std = fit_az.transform(cus_imp)
df_cus_std = pd.DataFrame(cus_std) # re-create dataframe

# applies PCA
cus_pca = pcafit.transform(df_cus_std)

# assigns points to clusters
cus_clust = kfit.predict(cus_pca)
```

### Step 3.3: Compare Customer Data to Demographics Data

At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.

Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.

Take a look at the following points in this step:

- Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
- *Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!*
- Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
- Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

```python
# kmean labels & add dataset with high NaN as cluster -1 
pop_clabels = kmeans.labels_
pop_clabels_nan = np.append(pop_clabels, [-1] * azdias_highNaN.shape[0])

# Create table with proportions
pop_freq = np.unique(pop_clabels_nan,return_counts=True)
pop_prop = pop_freq[1] / pop_freq[1].sum()
pop_prop

# customer dataset
# kmean labels & add dataset with high NaN as cluster -1 
cust_clabels = cus_clust
cust_clabels_nan = np.append(cust_clabels, [-1] * cus_highNaN.shape[0])

# Create table with proportions
cust_freq = np.unique(cust_clabels_nan,return_counts=True)
cust_prop = cust_freq[1] / cust_freq[1].sum()
```

```python
# plot customers Vs population
fig, ax = plt.subplots(figsize=(16, 10))
x_vals = np.append([-1], np.arange(kk))
bar_width = 0.35
opacity = 0.7

popu = plt.bar(x_vals, pop_prop, bar_width,
                 alpha=opacity,
                 color='k',
                 label='General Population')
 
custo = plt.bar(x_vals + bar_width, cust_prop, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Customer Data')

plt.xlabel('# of Cluster')
plt.ylabel('Proportion of cluster')
plt.title('Proportions of Clusters for General and Customer Data')
plt.xticks(x_vals + bar_width, x_vals)
plt.legend()
 
plt.tight_layout()
sns.set(style="whitegrid")
sns.despine();
```
<p align="center"><img src="/img/10.png?raw=true"/></p>

```python
# Plot for difference of proportions
fig, ax = plt.subplots(figsize=(16, 10))
diff = cust_prop - pop_prop
df_sign = pd.DataFrame(diff > 0, columns = ["pos"])
gen = plt.bar(x_vals, diff, bar_width, alpha=opacity, color=df_sign.pos.map({True: 'g', False: 'r'}))
plt.xlabel('Cluster number')
plt.ylabel('Difference in proportions')
plt.title('Difference in proportions between the population and customer Clusters')
plt.xticks(x_vals)
sns.despine();
```
<p align="center"><img src="/img/11.png?raw=true"/></p>

### What kinds of people are part of a cluster that is overrepresented in the customer data compared to the general population?

Some of the positive clusters are:
1. Cluster -1
2. Cluster 4
3. Cluster 7
4. Cluster 8

The kind of people who are over-represented in the customer data are very strongly correlated with clusters -1 and 8. They tend to have dependents at home and have a strained relationship with the internet. Judging by the group with a high number of missing observations (the '-1' cluster), they are 46-60 years old based on their names. They are also a fair-supplied type of energy consumers. They are financially prepared, and have a low sexual affinity. They are basically middle aged individuals on the verge of retirement.

### What kinds of people are part of a cluster that is underrepresented in the customer data compared to the general population?
Some of the negative clusters are:
1. Cluster 0
2. Cluster 3
3. Cluster 9
4. Cluster 11

The under-represented people are strongly correlated with clusters 3 and 11. They are the young professionals, who are financially prepared and not very religious. The are the more event-oriented and the more sensual minded. They hold higher degrees, do not own homes yet, and live in areas with like-minded career-oriented individuals.

### Discussion 3.3: Compare Customer Data to Demographics Data
Simply put, it seems like the client should focus more on young professionals since they are underrespresented in his customer base. While the older, almost retired, generation are relatively popular with the company, it needs to focus it's marketing efforts towards the young professionals. Since that is a section of the population it has yet to tap into. To achieve that, they must appeal to them through big events and potentially edgy advertising.