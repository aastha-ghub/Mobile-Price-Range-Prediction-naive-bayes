#!/usr/bin/env python
# coding: utf-8

#   <tr>
#         <td width="15%">
#         </td>
#         <td>
#             <div align="left">
#                 <font size=25px>
#                     <b>  Mobile Price Range Prediction
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>

# ## Problem Statement:
# Bob has started his own mobile company. He wants to give tough fight to big companies like Apple,Samsung etc.
# 
# He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he collects sales data of mobile phones of various companies.
# 
# Bob wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price. But he is not so good at Machine Learning. So he needs your help to solve this problem.
# 
# In this problem you do not have to predict actual price but a price range indicating how high the price is
# 

# ## Data Definition:
# 
# Input variables:
# 
# **Independent Variable**
# 
# 1. battery_power: Total energy a battery can store in one time measured in mAh
# 
# 2. blue: Has bluetooth or not
# 
# 3. clock_speed: speed at which microprocessor executes instructions
# 
# 4. dual_sim: Has dual sim support or not
# 
# 5. fc: Front Camera mega pixels
# 
# 6. four_g: Has 4G or not
# 
# 7. int_memory: Internal Memory in Gigabytes
# 
# 8. m_dep: Mobile Depth in cm
# 
# 9. mobile_wt: Weight of mobile phone
# 
# 10. n_cores: Number of cores of processor
# 
# 11. pc: Primary Camera mega pixels
# 
# 12. px_height: Pixel Resolution Height
# 
# 13. px_width: Pixel Resolution Width
# 
# 14. ram: Random Access Memory in Mega Bytes
# 
# 15. sc_h: Screen Height of mobile in cm
# 
# 16. sc_w: Screen Width of mobile in cm
# 
# 17. talk_time: longest time that a single battery charge will last when you are
# 
# 18. three_g: Has 3G or not
# 
# 19. touch_screen: Has touch screen or not
# 
# 20. wifi: Has wifi or not
# 
# **Dependent Variable**
# 
# 21. price_range: This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost).

# 1. **[Import Packages](#import_packages)**
# 2. **[Read Data](#Read_Data)**
# 3. **[Understand and Prepare the Data](#data_preparation)**
#     - 3.1 - [Data Types and Dimensions](#Data_Types)
#     - 3.2 - [Data Manipulation](#Data_Manipulation)
#     - 3.3 - [Missing Data Treatment](#Missing_Data_Treatment)
#     - 3.4 - [Statistical Summary](#Statistical_Summary)
# 4. **[EDA](#EDA)**    
#     - 4.1 - [Univariate Analysis](#Univariate_Analysis)
#     - 4.2 - [Bivariate Analysis](#Bivariate_Analysis)
#     - 4.3 - [Multivariate Analysis](#Multivariate_Analysis)
#     - 4.4 - [Conclusion of EDA](#Conclusion_of_EDA)
# 5. **[Label Encoding for categorical Variable](#Label_Encoding_for_categorical_Variable)**
# 6. **[Feature Selection](#feature_selection)**
# 7. **[Standardise Data](#Standardise_Data)**
# 8. **[ML Models](#ML_Models)**
#     - 8.1 - [Naive Bayes](#Naive_Bayes) 
#     
#   

# <a id='import_packages'></a>
# ## 1. Import Packages

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.model_selection import train_test_split
# Set default setting of seaborn
sns.set()


# <a id='Read_Data'></a>
# ## 2. Read the Data

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b>Read the data using read_csv() function from pandas<br> 
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[3]:


# read the data
raw_data = pd.read_csv('/Users/Admin/Downloads/Mobile Price Range Prediction/Dataset/mobile_price.csv')

# print the first five rows of the data
raw_data.head()


# In[4]:


data = raw_data.copy(deep = True)


# <a id='data_preparation'></a>
# ## 3. Understand and Prepare the Data

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b>The process of data preparation entails cleansing, structuring and integrating data to make it ready for analysis. <br><br>
#                         Here we will analyze and prepare data :<br>
#                         1. Check dimensions and data types of the dataframe <br>
#                         2. Data Manipulation<br>
#                         3. Check for missing values<br>
#                         4. Study summary statistics<br> 
#                                        </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='Data_Types'></a>
# ## 3.1 Data Types and Dimensions

# In[5]:


# get the shape
print(data.shape)


# **We see the dataframe has 21 columns and 2000 observations**

# In[6]:


# check the data types for variables
data.info()


# <table align='left'>
#     <tr>
#         <td width='8%'>
#             <img src='note.png'>
#         </td>
#         <td>
#             <div align='left', style='font-size:120%'>
#                     <b>From the above output, we see that all attributes are numerical, even the one which are
#                         categorical are represented as numeric</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='Data_Manipulation'></a>
# ## 3.2. Data Manipulation

# **Manipulating data types**

# In[7]:


data['n_cores'] = data['n_cores'].astype("object")
data['price_range'] = data['price_range'].astype("object")


# In[8]:


data.info()


# **Datatypes are updated**

# <a id='Missing_Data_Treatment'></a>
# ## 3.3. Missing Data Treatment

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b>In order to get the count of missing values in each column, we use the in-built function .isnull().sum()
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[9]:


# get the count of missing values
missing_values = data.isnull().sum()

# print the count of missing values
print(missing_values)


# There are no missing values present in the data.

# <a id='Statistical_Summary'></a>
# ## 3.4. Statistical Summary
# Here we take a look at the summary of each attribute. This includes the count, mean, the min and max values as well as some percentiles for numeric variables.

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b> In our dataset we have numerical variables. Now we check for summary statistics of all the variables<br>
#                         For numerical variables, we use .describe(). For categorical variables we use describe(include='object').
#           <br>
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[10]:


# data frame with numerical features
data.describe()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="note.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
# <b>The above output illustrates the summary statistics of all the numeric variables like the mean, median(50%), minimum, and maximum values, along with the standard deviation.</b>     </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[11]:


# data frame with categorical features
data.describe(include='object')


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="note.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
# <b>The above output illustrates the summary statistics of the categorical variables i.e n_cores(no.of cores in the mobile), predited variable price_range and the count of the majority level.</b>     </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='EDA'></a>
# ## 4. EDA

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b> Explore the features(independent variables) and check if they are inter-related to each other.<br><br>This is acheived using following steps:<br>
#                        1. Univariate Analysis<br>
#                        2. Bivariate Analysis<br>
#                        3. Multivariate Analysis<br>
#                 </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='Univariate_Analysis'></a>
# ## 4.1. Univariate Analysis

# **Exploring individual features**

# 4.1.1 Distribution of Battery Power

# In[12]:


data.battery_power.describe()


# In[13]:


sns.distplot(data.battery_power)


# **The above histogram shows that :**
# 
# * “battery power” attribute is almost symmetric in nature.
# * Minimum and Maximum battery power is 500 and 2000 respectively.
# * This dataset has fewer observations at the extreme values i.e minimum and maximum values
# * This indicates that people mainly buy mobile phones having an average battery power.
# 

# 4.1.2 Distribution of Bluetooth

# In[14]:


data.blue.value_counts()


# In[15]:


sns.countplot(data.blue,data=data,palette = "rainbow")
plt.show()


# **This countplot of “Blue” shows that:**
# 
# * With the plot we can say that there are only two types of responses for “blue”, ‘0’ which indicates phone do not have bluetooth and ‘1’ indicating, it has bluetooth service.
# * Both “0” and “1” are almost equal in proportion
# 

# 4.1.3 Distribution of Clock Speed

# In[16]:


data.clock_speed.describe()


# In[17]:


sns.distplot(data.clock_speed)


# **The above histogram shows that :**
# 
# * The Clock Speed ranges from 0.5 to 3 and is numeric
# * The dataset contain majority of phones having clock spped of “0.5”
# * Other clock speed’s are almost same in quantity
# 

# 4.1.4 Dual SIM

# In[18]:


data.dual_sim.value_counts()


# In[19]:


sns.countplot(data.dual_sim,data=data,palette = "rainbow")
plt.show()


# **This countplot of “dual_sim” shows that:**
# 
# * Dual sim has two responses “0” (no) and 1 (“yes”)
# * The feature is balanced, as they are almost equal in proportion
# 

# 4.1.5 Front Camera Megapixels

# In[20]:


data.fc.describe()


# In[21]:


sns.distplot(data.fc)


# In[22]:


data.fc[data.fc == 0].count()


# **The above histogram shows that :**
# 
# * It shows that most of the “fc” values are centered on 0 and count decreases as fc increases.
# * This explains that 474 phones in our data set do not have a front camera.
# * The data is right skewed i.e it is not symmetric
# 
# 

# Let's check skewness

# In[23]:


data.fc.skew()


# <table align='left'>
#     <tr>
#         <td width='8%'>
#             <img src='note.png'>
#         </td>
#         <td>
#             <div align='left', style='font-size:120%'>
#                     <b>Skew values discription : <br>
#                         1. A skewness value of 0 in the output denotes a symmetrical distribution of values<br>
#                         2. A negative skewness value in the output indicates an asymmetry in the distribution and the tail is larger towards the left hand side of the distribution.<br>
#                         3. A positive skewness value in the output indicates an asymmetry in the distribution and the tail is larger towards the right hand side of the distribution.<br>
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# 4.1.6 four_g

# In[24]:


data.four_g.value_counts()


# In[25]:


sns.countplot(data.four_g,data=data,palette = "rainbow")
plt.show()


# **This countplot of “four_g” shows that:**
# 
# * There are 2 unique categories present in the four_g attribute, 0(“no”) and 1(“yes”).
# * Most of the phones have four_g service i.e 52.15%
# * There is no imbalance in the categories of four_g attribute.
# 
# 

# 4.1.7 int_memory

# In[26]:


data.int_memory.describe()


# In[27]:


sns.distplot(data.int_memory)


# **The above distribution shows that :**
# 
# * The internal memory ranges from 2 to 64 GB
# * Also, the dataset have phone’s of different generation.i.e Older mobiles to latest one’s.
# 

# 4.1.8 m_dep

# In[28]:


data.m_dep.describe()


# In[29]:


sns.distplot(data.m_dep)


# **The above distribution shows that :**
# 
# * We can classify phones into three category based on m_dep into “low” , “mid” and “high”
# * The attribute ranges from 0.1 cm to 1 cm
# 

# 4.1.9 mobile_wt

# In[30]:


data.mobile_wt.describe()


# In[31]:


sns.distplot(data.mobile_wt)


# **The above distribution shows that :**
# 
# * The mobile weight is distributed from 80 gm to 200 gm
# * The extreme values have minimum count
# 

# 4.1.10 n_cores

# In[32]:


data.n_cores.value_counts()


# In[33]:


sns.countplot(data.n_cores,data=data,palette = "rainbow")
plt.show()


# **This countplot of “n_cores” shows that:**
# 
# * There are 8 unique categories in the n_cores attribute.
# * All this categories are almost equal in quantity except mobiles with 6 and 4 cores.
# 
# 

# 4.1.11 pc

# In[34]:


data.pc.describe()


# In[35]:


sns.distplot(data.pc)


# **The above distribution shows that :**
# 
# * The pc feature ranges from 0 to 20
# * The statistical summary explains that there are no outliears.
# * The feature is not symmetric
# 
# 

# 4.1.12 px_height

# In[36]:


data.px_height.describe()


# In[37]:


sns.distplot(data.px_height)


# **The above distribution shows that :**
# 
# * Pixel resolution height ranges from 0 to 1960
# * The feature is skewed to right i.e not symmetric.
# 
# 

# 4.1.13 px_width

# In[38]:


data.px_width.describe()


# In[39]:


sns.distplot(data.px_width)


# **The above distribution shows that :**
# 
# * The feature ranges from 500 to 1998
# * Also data is concentrated towards the center
# 
# 

# 4.1.14 ram

# In[40]:


data.ram.describe()


# In[41]:


sns.distplot(data.ram)


# **The above distribution shows that :**
# 
# * The attribute ranges from 256 MB to 3998 MB
# * There extremes are less in quantity
# 
# 

# 4.1.15 sc_h

# In[42]:


data.sc_h.describe()


# In[43]:


sns.distplot(data.sc_h)


# **The above distribution shows that :**
# 
# * Majoirty of phones have screen height as 193 , 157 , 151 cm’s
# * The feature ranges from 5 cm to 19 cm
# 
# 

# 4.1.16 sc_w

# In[44]:


data.sc_w.describe()


# In[45]:


sns.distplot(data.sc_w)


# **The above distribution shows that :**
# 
# * The attribute ranges from 0 cm to 18 cm
# * The screen width of 0 is clearly a outliear
# 
# 

# 4.1.17 talk_time

# In[46]:


data.talk_time.describe()


# In[47]:


sns.distplot(data.talk_time)


# **The above distribution shows that :**
# 
# * Majoirty of phones have talk time of 6 ,19,15,16,4,7 hours.
# * The feature ranges from 2 hours to 19 hours
# 
# 

# 4.1.18 three_g

# In[48]:


data.three_g.value_counts()


# In[49]:


sns.countplot(data.three_g,data=data,palette = "rainbow")
plt.show()


# **This barchart says that:**
# 
# * The attribute has two category 0 (no) and 1 (yes)
# * Majority of the phones have 3G service.

# 4.1.19 touch_screen

# In[50]:


data.touch_screen.value_counts()


# In[51]:


sns.countplot(data.touch_screen,data=data,palette = "rainbow")
plt.show()


# **This countplot of “touch_screen” shows that:**
# 
# * The feature is balanced
# * The dataset equally focuses on phones having touch screen or buttons
# 

# 4.1.20 wifi

# In[52]:


data.wifi.value_counts()


# In[53]:


sns.countplot(data.wifi,data=data,palette = "rainbow")
plt.show()


# **This countplot of “wifi” shows that:**
# 
# * The attribute has two category 0 (no) and 1 (yes)
# * The feature is balanced
# 

# 4.1.21 price_range

# In[54]:


data.price_range.value_counts()


# In[55]:


sns.countplot(data.price_range,data=data,palette = "rainbow")
plt.show()


# **Dependent variable is uniformly distributed**

# <a id='Bivariate_Analysis'></a>
# ## 4.2. Bivariate Analysis

# 4.2.1 Battery power

# In[56]:


sns.boxplot(y = data.battery_power , x = data.price_range )


# 4.2.2 blue

# In[57]:


sns.countplot(data.price_range,hue=data.blue)


# **Majority of phones of price range from 0 to 2 dont have bluetooth on other hand price range of 3 have bluetooth service**

# 4.2.3 clock_speed

# In[58]:


sns.boxplot(y = data.clock_speed , x = data.price_range )


# 4.2.4 dual_sim

# In[59]:


sns.countplot(data.price_range,hue=data.dual_sim)


# **Majority of phones have dual sim service.**

# 4.2.5 fc

# In[60]:


sns.boxplot(y = data.fc , x = data.price_range )


# **There are few outliears in fc**

# 4.2.6 four_g

# In[61]:


sns.countplot(data.price_range,hue=data.four_g)


# **Majority of phones of only price range 2 dont have 4G service.**

# 4.2.7 int_memory

# In[62]:


sns.boxplot(y = data.int_memory , x = data.price_range )


# 4.2.8 m_dep

# In[63]:


sns.boxplot(y = data.m_dep , x = data.price_range )


# 4.2.9 mobile_wt

# In[64]:


sns.boxplot(y = data.mobile_wt , x = data.price_range )


# 4.2.10 n_cores
# 

# In[65]:


sns.countplot(data.price_range,hue=data.n_cores)


# * Price range 0 has majority of phones with 2 core processors
# * Price range 1 has majority of phones with 4 core processors
# * Price range 2 has majority of phones with 4 core processors
# * Price range 3 has majority of phones with 5 and 7 core processors

# 4.2.11 pc

# In[66]:


sns.boxplot(y = data.pc , x = data.price_range )


# 4.2.12 px_height

# In[67]:


sns.boxplot(y = data.px_height , x = data.price_range )


# 4.2.13 px_width

# In[68]:


sns.boxplot(y = data.px_width , x = data.price_range )


# 4.2.14 ram

# In[69]:


sns.boxplot(y = data.ram , x = data.price_range )


# **There are few outliears**

# 4.2.15 sc_h

# In[70]:


sns.boxplot(y = data.sc_h , x = data.price_range )


# 4.2.16 sc_w

# In[71]:


sns.boxplot(y = data.sc_w , x = data.price_range )


# 4.2.17 talktime

# In[72]:


sns.boxplot(y = data.talk_time , x = data.price_range )


# 4.2.18 three_g

# In[73]:


sns.countplot(data.price_range,hue=data.three_g)


# **Majority of phones irrespective of price range have 3G service**

# 4.2.19 touch_screen

# In[74]:


sns.countplot(data.price_range,hue=data.touch_screen)


# **From countplot we can conclude that :**
# 
# * Phones of price range 0 and 1 have majority of touch screen service.
# * Majority of phones of price range 2 and 3 do not have touch screen service.
# 
# 

# 4.2.20 wifi

# In[75]:


sns.countplot(data.price_range,hue=data.wifi)


# **Wifi service is almost unbiased.**

# <a id='Multivariate_Analysis'></a>
# ## 4.3. Multivariate Analysis

# 4.3.1 Correlation among the features

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b> Call the corr() function which will return the correlation matrix of numeric variables</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[76]:


# check correlation
data_num = data.copy(deep = True)
data_num.price_range = data.price_range.astype("int64")
corr = data_num.corr()
corr


# In[77]:


#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data_num.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[78]:


plt.figure(figsize=(30, 15))
sns.heatmap(corr[(corr >= 0.9) | (corr <= -0.9)], 
            cmap='YlGnBu', vmax=1.0, vmin=-1.0,
            annot=True, annot_kws={"size": 15})
plt.title('Correlation between features', fontsize=15)
plt.show()


# **Conclusion from heatmap :**
# 
# * As we can see our target price range has highly positive correlation between ram.
# * Below features have highly positive correlation
#     * 3G and 4G
#     * pc(Primary Camera mega pixels) and fc(Front Camera mega pixels)
#     * px_weight(Pixel Resolution Width) and px_height(Pixel Resolution Height)
#     * sc_w(Screen Width of mobile in cm) and sc_h(Screen Height of mobile in cm)

# <a id='Conclusion_of_EDA'></a>
# ## 4.4 Conclusion of EDA

# **Feature Removal:**
# 
# pc(Primary Camera mega pixels) and fc(Front Camera mega pixels) are highly correlated
# 
# **Outliers Summary:**
# 
# * fc
# * px_heigth
# * ram
# * sc_w
# 
# **Note : we will not remove outliears are they are outliears for individual price range, so ignoring them**
# 

# <a id='Label_Encoding_for_categorical_Variable'></a>
# ## 5. Label Encoding for categorical Variable

# In[79]:


data_with_dummies = pd.get_dummies(data.drop(['price_range'],axis = 1),drop_first=True)
data_with_dummies['price_range'] = data.price_range
data_with_dummies.head()


# <a id='feature_selection'></a>
# ## 6. Feature Selection

# In[80]:


# Split the data into 40% test and 60% training
X = data_with_dummies.drop(['price_range'],axis = 1)
y = data_with_dummies['price_range'].astype('int64')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


# In[81]:


from sklearn.ensemble import RandomForestClassifier
# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the classifier
clf.fit(X_train, y_train)


# In[82]:


feat_labels = X.columns.values
# Print the name and gini importance of each feature
feature_importance = []
for feature in zip(feat_labels, clf.feature_importances_):
    #rint(feature)
    feature_importance.append(feature)


# In[83]:


feature_importance


# In[84]:


from sklearn.feature_selection import SelectFromModel
# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.01
sfm = SelectFromModel(clf, threshold=0.01)


# In[85]:


# Train the selector
sfm.fit(X_train, y_train)


# In[86]:


selected_features = []
# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    selected_features.append(feat_labels[feature_list_index])


# In[87]:


selected_features


# In[88]:


data_selected = data_with_dummies[selected_features]
data_selected.head()


# <a id="Standardise_Data"> </a>
# ## 7. Standardise Data

# In[89]:


from sklearn.preprocessing import MinMaxScaler


# In[90]:


scaler = MinMaxScaler()


# In[91]:


scaler.fit(data_selected)


# In[92]:


data_standardised = scaler.fit_transform(data_selected)


# **split data into train and test**

# In[93]:


from sklearn.model_selection import train_test_split
# let us now split the dataset into train & test
X = data_standardised
y = data_with_dummies['price_range'].astype('int64')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)

# print the shape of 'x_train'
print("X_train ",X_train.shape)

# print the shape of 'x_test'
print("X_test ",X_test.shape)

# print the shape of 'y_train'
print("y_train ",y_train.shape)

# print the shape of 'y_test'
print("y_test ",y_test.shape)


# <a id="ML_Models"> </a>
# ## 8. ML Models

# <a id="Naive_Bayes"> </a>
# ## 8.1 Naive Bayes

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b> Now we build a ensemble model using Naive Bayes. We start with our data set gradually proceeding with our analysis<br><br>
#                         In order to build a  ensemble model using Naive Bayes, we do the following:<br>
#                         1. Build the model<br>
#                         2. Predict the values<br>
#                         3. Compute the accuracy measures<br>
#                         4. Tabulate the results <br>
#                       </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. Build the model**

# In[94]:


from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
# build the model
gnb = GaussianNB()

# define the ovr strategy
GNB = OneVsRestClassifier(gnb)

# fit the model
GNB.fit(X_train, y_train)


# **2. Predict the values**

# In[95]:


# predict the values
y_pred_GNB = GNB.predict(X_test)


# **3. Compute accuracy measures**

# In[96]:


from sklearn.metrics import confusion_matrix
# compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_GNB)

# label the confusion matrix  
conf_matrix = pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1','Predicted:2','Predicted:3'],index=['Actual:0','Actual:1','Actual:2','Actual:3'])

# set sizeof the plot
plt.figure(figsize = (8,5))

# plot a heatmap
# cmap: colour code used for plotting
# annot: prints the correlation values in the chart
# annot_kws: sets the font size of the annotation
# cbar=False: Whether to draw a colorbar
# fmt: string formatting code to use when adding annotations
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu", cbar=False)
plt.show()


# In[97]:


from sklearn.metrics import classification_report
# accuracy measures by classification_report()
result = classification_report(y_test,y_pred_GNB)

# print the result
print(result)


# **4. Tabulate the results**

# In[98]:


from sklearn import metrics
# create the result table for all accuracy scores
# Accuracy measures considered for model comparision are 'Model', 'AUC Score', 'Precision Score', 'Recall Score','Accuracy Score','Kappa Score', 'f1 - score'

# create a list of column names
cols = ['Model', 'Precision Score', 'Recall Score','Accuracy Score','f1-score']

# creating an empty dataframe of the colums
result_tabulation = pd.DataFrame(columns = cols)

# compiling the required information
Naive_bayes = pd.Series({'Model': "Naive Bayes",
                 'Precision Score': metrics.precision_score(y_test, y_pred_GNB,average="macro"),
                 'Recall Score': metrics.recall_score(y_test, y_pred_GNB ,average="macro"),
                 'Accuracy Score': metrics.accuracy_score(y_test, y_pred_GNB),
                  'f1-score':metrics.f1_score(y_test, y_pred_GNB,average = "macro")})



# appending our result table
result_tabulation = result_tabulation.append(Naive_bayes , ignore_index = True)

# view the result table
result_tabulation


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="note.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b>It can be seen from the above result that accuracy measures for the Naive Bayes is 79.66 %</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:





# In[ ]:




