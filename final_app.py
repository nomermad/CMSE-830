import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import plotly.express as px
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression # this could be any ML method
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
import plotly.figure_factory as ff
import zipfile

st.title("Demographic and Economic Patterns in Homicide Incidents")
st.write('Welcome to my Streamlit app!')# Load and display your data, add widgets, etc.

@st.cache_data
def load_data():
    zip_file_path = 'database.csv.zip'

    with zipfile.ZipFile(zip_file_path, 'r') as z:

        file_names = z.namelist()
        

        with z.open('database.csv') as csv_file:
            data = pd.read_csv(csv_file)
    
    return data


data = load_data()
data.replace("Unknown", np.nan, inplace=True)

us_unemployment = pd.read_csv('unemployment_data_us.csv')

filtered_violence = data.rename(columns={"Year": "year"})
month_mapping = {
    'January': 1, 'Febuary': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

filtered_violence['month'] = filtered_violence['Month'].map(month_mapping)
filtered_violence.drop(columns=['Month'], inplace=True)
filtered_unemployment = us_unemployment.rename(columns={"Year": "year"})
month_mapping = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}
filtered_unemployment['month'] = filtered_unemployment['Month'].map(month_mapping)
filtered_unemployment.drop(columns=['Month'], inplace=True)
merged_data = pd.merge(filtered_violence, filtered_unemployment, on=['year','month'])
merged_data.drop_duplicates()
merged_data = merged_data.drop(['Record ID', 'Agency Code','City','Incident','Agency Name','Agency Type','Record Source'], axis=1)

data = merged_data.copy() #creating a copy of the dataset 


X = data[['year', 'High_School', 'Associates_Degree', 'Professional_Degree', 'White', 
          'Black', 'Asian', 'Hispanic', 'Men', 'Women']]
y = data['Primary_School']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)


mice_imputer = IterativeImputer(random_state=42, max_iter=50)
X_train_mice = pd.DataFrame(mice_imputer.fit_transform(X_train_scaled), columns=X_train.columns, index=X_train.index)
X_test_mice = pd.DataFrame(mice_imputer.transform(X_test_scaled), columns=X_test.columns, index=X_test.index)

lr_mice = LinearRegression()
lr_mice.fit(X_train_mice, y_train)

y_pred_mice = lr_mice.predict(X_test_mice)
mse_mice = mean_squared_error(y_test, y_pred_mice)
r2_mice = r2_score(y_test, y_pred_mice)



mean_imputer = SimpleImputer(strategy='mean')
X_train_mean = pd.DataFrame(mean_imputer.fit_transform(X_train_scaled), columns=X_train.columns, index=X_train.index)
X_test_mean = pd.DataFrame(mean_imputer.transform(X_test_scaled), columns=X_test.columns, index=X_test.index)


lr_mean = LinearRegression()
lr_mean.fit(X_train_mean, y_train)


y_pred_mean = lr_mean.predict(X_test_mean)
mse_mean = mean_squared_error(y_test, y_pred_mean)
r2_mean = r2_score(y_test, y_pred_mean)

numeric_columns = merged_data.select_dtypes(include=[np.number]).columns
homicide_numeric = merged_data[numeric_columns]


data_with_missing = homicide_numeric[homicide_numeric.isnull().any(axis=1)]
homicide_without_missing = homicide_numeric.dropna()

scaler = StandardScaler()
homicide_scaled = pd.DataFrame(scaler.fit_transform(homicide_without_missing), columns=homicide_without_missing.columns)

# initialize and fit KNN imputer
imputer = KNNImputer(n_neighbors=5)
imputer.fit(homicide_scaled)

# function to impute and inverse transform the data
def impute_and_inverse_transform(data):
    # Ensure 'data' is always a DataFrame with proper column names
    scaled_data = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)
    imputed_scaled = imputer.transform(scaled_data)
    return pd.DataFrame(scaler.inverse_transform(imputed_scaled), columns=data.columns, index=data.index)

# impute missing values
homicide_imputed = impute_and_inverse_transform(homicide_numeric)

# Update the original dataset with the imputed numerical data
merged_data[numeric_columns] = homicide_imputed

#using simple input for categorical variables
categorical_columns = ['Crime Type', 'Perpetrator Sex', 'Victim Sex','Perpetrator Race','Perpetrator Ethnicity' ,'Victim Race'
                       ,'Victim Ethnicity','Relationship','Weapon']
imputer = SimpleImputer(strategy='most_frequent')
merged_data[categorical_columns] = imputer.fit_transform(merged_data[categorical_columns])


merged_data['Perpetrator Age'] = pd.to_numeric(merged_data['Perpetrator Age'], errors='coerce').astype('Int64')
merged_data = merged_data[(merged_data['Victim Age'] >= 18.0)]
merged_data = merged_data[(merged_data['Perpetrator Age'] >= 18.0)]

unemployment_columns = ['Primary_School', 'High_School', 'Associates_Degree', 
                        'Professional_Degree', 'White', 'Black', 'Asian', 
                        'Hispanic', 'Men', 'Women']

merged_data['Overall_Unemployment_Rate'] = merged_data[unemployment_columns].mean(axis=1)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
le = LabelEncoder()
merged_data['Victim_Sex_encoded'] = le.fit_transform(merged_data['Victim Sex'])
merged_data['Perpetrator_Sex_encoded'] = le.fit_transform(merged_data['Perpetrator Sex'])
merged_data['Victim_Race_encoded'] = le.fit_transform(merged_data['Victim Race'])
merged_data['Perpetrator_Race_encoded'] = le.fit_transform(merged_data['Perpetrator Race'])
merged_data['Perpetrator_Ethnicity_encoded'] = le.fit_transform(merged_data['Perpetrator Ethnicity'])
merged_data['Victim_Ethnicity_encoded'] = le.fit_transform(merged_data['Victim Ethnicity'])

white_mean = merged_data['White'].mean()
black_mean = merged_data['Black'].mean()
hispanic_mean = merged_data['Hispanic'].mean()
asian_mean = merged_data['Asian'].mean()
men_mean = merged_data['Men'].mean()
women_mean = merged_data['Women'].mean()

unemployment_min = merged_data['Overall_Unemployment_Rate'].min()
unemployment_max = merged_data['Overall_Unemployment_Rate'].max()

victim_race_counts = merged_data['Victim Race'].value_counts()
victim_gender_counts = merged_data['Victim Sex'].value_counts()
perpetrator_race_counts = merged_data['Perpetrator Race'].value_counts()
perpetrator_gender_counts = merged_data['Perpetrator Sex'].value_counts()
perpetrator_ethnicity_count = merged_data['Perpetrator Ethnicity'].value_counts()
victim_ethnicity_count = merged_data['Victim Ethnicity'].value_counts()

merged_data['Unemployment_Rate_Category'] = 0
merged_data.loc[merged_data['Overall_Unemployment_Rate'] < 5, 'Unemployment_Rate_Category'] = 0
merged_data.loc[(merged_data['Overall_Unemployment_Rate'] >= 5) & (merged_data['Overall_Unemployment_Rate'] <= 10), 'Unemployment_Rate_Category'] = 1
merged_data.loc[merged_data['Overall_Unemployment_Rate'] > 10, 'Unemployment_Rate_Category'] = 2



st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a section", ["Introduction","Data Overview", "Data Analysis", "Visualizations"])

if option == "Introduction":
    st.title("How to use my Streamlit app!")
    st.markdown(""" My streamlit app contains a lot of useful information and includes a lot of interactive features to do so. To start, the app is 
    composed of three tabs on the navigation bar, which has data processing, exploratory data analysis, and visualization. Under the data 
    processing tab, you can view the homicide data and the unemployment data. There is an interactive drop down menu, where you can choose what 
    table you want to see. Then, the missing values of both tables were put into a heatmap, which is interactive as well and you can use the drop 
    down menu to choose what table you want to see. To move to the exploratory data analysis page, you can select it from the drop down menu. The 
    first interactive feature on this page is choosing to look at specific features in a table and comparing those features. The mean value was 
    found for relevant columns, and you can click the checkbox to display which mean values you would like to look at. Finally, you can see the 
    visualizations by clicking on the visualization tab. 6 graphs were created, and you can click the drop down button to choose which graph you 
    want to view. """)


elif option == "Data Overview":
    st.title("Summary of Data Collection and Preparation")
    st.markdown("""
    ### Summary of Data Collection and Preparation 

    * **Datasets**: For my datasets, I found two different datasets. The first dataset is from Kaggle and has data about homicides in the United 
    States starting in 1980. The second dataset I found on Kaggle as well, and that dataset focused on unemployment rates in the United States 
    starting in 2010.
    * **Preparation:** After loading in my datasets, I used `.head()` and `.describe()` to see if there were any missing variables. I found that in 
    the homicide dataset, there were no missing numerical values, but there were categorical variables that had "Unknown" as values. I converted 
    the unknown values to NaN values using NumPy. After using `.describe()` on the unemployment dataset, I noticed each numerical column was 
    missing about 10 data points.
    * **Handling missing data:** After noticing there were missing data points, I used a heatmap to show where all the missing variables were. 
    After that, I used two different methods to impute values where the missing values were. I compared the MICE and mean imputation, and then went 
    with the KNN imputation, as I thought it did a better job.
    * **Overall Unemployment Column:** A column called "Overall Unemployment Rate" was created to display the average unemployment using the 
    columns from the unemployment dataset for each year. This could allow for better analysis.
    * **Masking the Data:** Since only adults are typically employed, I thought it made the most sense to filter the age to only individuals older 
    than 18.
    * **Encoding Categorical Columns:** I encoded the columns "Perpetrator Sex", "Perpetrator Race", "Victim Sex", and "Victim Race". If the 
    perpetrator's sex is male, the column takes a 1, and a 0 if the perpetrator's sex is female. If the victim's sex is female, it takes a 0, and a 
    1 if the victim's sex is male. In both datasets, the months were changed from January-December to 1-12, to allow for easier analysis. """)
    
    selection_initial_table = st.selectbox("Choose a table to display:", 
                           ["Initial Homicide Table",
                            "Initial Unemployment Table"])
    if selection_initial_table == "Initial Homicide Table":
        st.write("Here is a preview of the Homicde data:")
        st.dataframe(data.head())

    elif selection_initial_table == "Initial Unemployment Table":
        st.write("Here is a preview of the Unemployment data:")
        st.dataframe(us_unemployment.head()) 

    st.write("Here is a preview of the Homicide data replacing the unknown values with Na values")
    st.dataframe(data.head())

    selection_describing_dataset = st.selectbox("Choose a table to display:", 
                                                ["Describing Homicide Table",
                                                 "Describing Unemployment Table"])

    if selection_describing_dataset == "Describing Homicide Table":
        st.write("Checking the Homicide Dataset:")
        st.dataframe(data.describe())

    elif selection_describing_dataset == "Describing Unemployment Table":
        st.write("Checking the Unemployment data:")
        st.dataframe(us_unemployment.describe()) 

    selection_missingvalues_heatmap = st.selectbox("Choose a graph to display:", 
                               ["Heatmap of Missing Values in Homicide Dataset",
                                "Heatmap of Missing Values in Unemployment Dataset"])

    if selection_missingvalues_heatmap == "Heatmap of Missing Values in Homicide Dataset":
        missing_values = data.isnull() 
        plt.figure(figsize=(10, 10))
        sns.heatmap(missing_values, cbar=False, cmap='viridis', yticklabels=False)
        plt.title('Heatmap of Missing Values in Homicide Dataset', fontsize=12)
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        st.pyplot(plt.gcf()) 

    elif selection_missingvalues_heatmap == "Heatmap of Missing Values in Unemployment Dataset":
        missing_values = us_unemployment.isnull() 
        plt.figure(figsize=(10, 10))
        sns.heatmap(missing_values, cbar=False, cmap='viridis', yticklabels=False)
        plt.title('Heatmap of Missing Values in Unemployment Dataset', fontsize=12)
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        st.pyplot(plt.gcf()) 

    st.write('Checking what the columns are')
    st.dataframe(merged_data.columns)

    st.write(f"MICE Imputation Results: Mean Squared Error: {mse_mice:.4f}, R2 Score: {r2_mice:.4f}")
    st.write(f"Mean Imputation Results: Mean Squared Error: {mse_mean:.4f}, R2 Score: {r2_mean:.4f}")

    missing_values = merged_data.isnull()

    st.write('Checking that all the missing values are gone')
    plt.figure(figsize=(10, 10))
    sns.heatmap(missing_values, cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Heatmap of Missing Values in Homicide Dataset', fontsize=12)
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()
    st.pyplot(plt.gcf())



elif option == "Data Analysis":
    st.title("Summary of Data Analysis")
    st.markdown(""" 
    * **Statistics for specific Columns:** The mean unemployment for each demographic column in the unemployment dataset was calculated, to get a 
    general idea about those columns. 
    * **What types of Variables are there:** There is a mix of categorical and numerical columns. The categorical columns are state, crime typed, 
    crime solved, victim sex, victim race, victim ethnicity, perpetrator sex, perpetrator race, perpetrator ethnicity, relationship and weapon 
    used. The rest of the columns were numerical. 
    * **Value Counts:** The value counts for victim sex, victim race, perpetrator sex, and perpetrator race were explored. I used value counts for 
    these variables, as it did not make sense to explore the mean
    * **Correlation Graph:** A correlation matrix was created using relevant features. Upon looking at the correlation, there was almost no 
    correlation between the selected features from the unemployment dataframe and the homicide dataframe.
    * **Z-Score Outliers:** The columns were scaled using the z-score and the outliers were printed, which were how many rows were less than -3 or 
    greater than 3 after scaling. The columns with outliers were Victim Age with 125 outliers, Perpetrator Age with 458 outliers, Victim Count with 
    1016 outliers, Perpetrator Count with 1139 outliers, perpetrator ethnicity with 3807 outliers, Unemployment Rate Category with 3610 outliers, 
    and Perpetrator_Sex_encoded with 3230 outliers.
    * **Masking:** A new column was created that assigned a category for the unemployment rates. 0%-5% is low, 5%-10% is middle, and anything 
    greater than 10% is a high unemployment rate. 
    * **Figure 1:** Figure one is a plot distribution that shows the perpetrator's sex, with the range/category of unemployment they may fall 
    under. A majority of perpetrators are males that fall under the middle unemployment category. 
    * **Figure 2:** Figure two illustrates a multi dimensional plot showing the relationship between year, overall unemployment rate and 
    perpetrator sex.
    * **Figure 3:** Figure three illustrates the relationship of the overall unemployment rate and the unemployment rates based on degree. From the 
    scatter plot, it can be conveyed that those with more schooling have a lower unemployment rate, as those who only completed highschool have a 
    higher unemployment rate compared to those with a professional degree.
    * **Figure 4:** Figure four conveys the relationship of victim count with the overall unemployment rate, coloring by year. The year 2014 had 
    lower unemployment rates compared to the year 2010. 
    * **Figure 5:** Figure five conveys the relationship of the encoded victim sex and the overall unemployment rate using a kde plot. From the 
    plot, there is a dark center around 1 on the x axis and 10. The darker points are where more points lie, which suggest that a majority of the 
    victims sex are males, with a higher unemployment rate. 
    * **Figure 6:** Figure six conveys the relationship of average homicide victim count by state and unemployment rate. It is a bar graph that has 
    states on the x axis and the average victim count for each state on the y axis. The bar graph is colored by unemployment. Alabama had the 
    highest unemployment rate, but had the lowest average victim count.
    * **What I got from the Data/What was I hoping to find:** What I was hoping to find was if there was a relationship between patterns in 
    unemployment and homicide trends. 
    * **What my overall results show:** There is no overall impact of unemployment rates on homicide trends, however a majority of perpetrators are 
    male. The kde visualization suggests that a majority of the data points are male perpetrators with a middle/high unemployment rate. Ultimately, 
    there is a relationship between the sex of an individual a
    * **Ways to Improve:** One way I could have improved the project was focusing on class distributions. There may not have been equal class 
    sizes, which could have impacted the results I got. My findings suggest that other factors, such as social or cultural factors, may play a 
    stronger role in homicide incidents. Moreover, this gives an opportunity for further exploration of the data, by potentially exploring 
    different patterns. The findings also suggest that there may be a relationship between the data, but it could be a more complex or nonlinear 
    relationship. Polynomial regression or logistic regression may capture a better relationship. The homicide dataset was also missing a 
    significant amount of categorical data points. While simple input was used to fill in the missing points, that could have also contributed to 
    the relationship and finding a lack of correlation. Furthermore, only specific categorical variables were encoded and looked at, and encoding 
    those and using different features such as if the crime was unsolved, weapon type, or the relationship between the perpetrator and victim may 
    have changed the correlation""")


    selected_features = st.multiselect("Select Features to Display for the merged Dataset", options=merged_data.columns.tolist())
    if selected_features:
        st.write(merged_data[selected_features])

    if st.checkbox("Show average unemployment rate for White Americans"):
        st.write("The average unemployment rate for White Americans is", white_mean)

    if st.checkbox("Show average unemployment rate for Black Americans"):
        st.write("The average unemployment rate for Black Americans is", black_mean)

    if st.checkbox("Show average unemployment rate for Hispanic Americans"):
        st.write("The average unemployment rate for Hispanic Americans is", hispanic_mean)

    if st.checkbox("Show average unemployment rate for Asian Americans"):
        st.write("The average unemployment rate for Asian Americans is", asian_mean)

    if st.checkbox("Show average unemployment rate for Men"):
        st.write("The average unemployment rate for Men is", men_mean)

    if st.checkbox("Show average unemployment rate for Women"):
        st.write("The average unemployment rate for Women is", women_mean)

    if st.checkbox("Show minimum American Overall Unemployment Rate"):
        st.write("The min unemployment rate is",unemployment_min)

    if st.checkbox("Show maximum American Overall Unemployment Rate"):
        st.write("The max unemployment rate is",unemployment_max)

    st.write(victim_race_counts)
    st.write(victim_gender_counts)
    st.write(perpetrator_gender_counts)
    st.write(perpetrator_race_counts)
    st.write(perpetrator_ethnicity_count)
    st.write(victim_ethnicity_count)
    st.write(merged_data[['White', 'Black', 'Hispanic', 'Men', 'Women']].describe())

    selected_features = ['year', 'Overall_Unemployment_Rate', 'Victim Count', 
                         'Victim_Sex_encoded', 'Perpetrator_Sex_encoded']

    correlation_matrix = merged_data[selected_features].corr().values

    fig_corr = ff.create_annotated_heatmap(z=correlation_matrix, x=selected_features, y=selected_features)
    st.write("Correlation matrix with selected features")
    st.plotly_chart(fig_corr)

    victim_age_by_sex = merged_data.groupby('Victim Sex')['Victim Age'].mean()
    st.dataframe(victim_age_by_sex)


    numeric_cols = merged_data.select_dtypes(include=[np.number]).columns#This line selects all the numeric columns from the dataset

    merged_data_zscore = merged_data[numeric_cols].apply(zscore)#This line applies z-score scaling to the numeric columns. This scales the data so each column has a mean of 0 and a standard deviation of 1(I am pretty sure based on Tuesday's lecture). 

    for i in numeric_cols:
    # Outliers are values where the z-score is less than -3 or greater than 3
        outliers = merged_data_zscore[(merged_data_zscore[i] < -3) | (merged_data_zscore[i] > 3)]
    
    # Print the column name and the number of outliers
        st.write("The z-score of every column")
        st.write(f"Column: {i}, Number of outliers: {len(outliers)}")




else:
    st.write("Show visualizations here.")
    selection_graph_eda = st.selectbox("Choose a graph to display:", 
                           ["Histogram of Perpetrator Sex and Unemployment Rate Category",
                            "3D Scatter Plot of Year, Overall Unemployment Rate and Perpetrator Sex",
                            "Unemployment Rates based on Highest Degree",
                            "Homicide Rates vs. Unemployment Rates",
                            "2D KDE Plot of Perpetrator Sex and Unemployment Rate",
                            "Average Homicide Victim Count by State"])

# Based on the selected option, render the corresponding graph
    if selection_graph_eda == "Histogram of Perpetrator Sex and Unemployment Rate Category":
        fig1 = px.histogram(merged_data, 
                            x='Perpetrator Sex', 
                            color='Unemployment_Rate_Category',  
                            title='Distribution of Unemployment Rate by Category', 
                            barmode='group',
                            labels={'Perpetrator Sex': 'Perpetrator Sex'})
        st.plotly_chart(fig1)

    elif selection_graph_eda == "3D Scatter Plot of Year, Overall Unemployment Rate and Perpetrator Sex":
        fig2 = px.scatter_3d(merged_data, 
                             x='year', 
                             y='Overall_Unemployment_Rate', 
                             z='Perpetrator_Sex_encoded', 
                             title='3D Scatter Plot',
                             labels={'year':'Year', 
                                 'Overall_Unemployment_Rate': 'Unemployment Rate (%)', 
                                 'Perpetrator_Sex_encoded':'Perpetrator Sex'},
                             color="Victim Count")
        st.plotly_chart(fig2)

    elif selection_graph_eda == "Unemployment Rates based on Highest Degree":
        fig3 = px.scatter(merged_data, 
                          x='Overall_Unemployment_Rate', 
                          y=['High_School','Associates_Degree','Professional_Degree'], 
                          labels={'Overall_Unemployment_Rate': 'Overall Unemployment Rate (%)'},
                          title='Unemployment Rates based on Degree')
        st.plotly_chart(fig3)

    elif selection_graph_eda == "Homicide Rates vs. Unemployment Rates":
        fig4 = px.scatter(merged_data, 
                          x='Victim Count', 
                          y='Overall_Unemployment_Rate',
                          color='year', 
                          size='year',
                          title='Homicide Rates vs. Overall Unemployment Rates',
                          labels={'Overall_Unemployment_Rate': 'Overall Unemployment Rate (%)', 
                              'Victim Count': 'Number of Homicide Victims'})
        st.plotly_chart(fig4)

    elif selection_graph_eda == "2D KDE Plot of Perpetrator Sex and Unemployment Rate":
        plt.figure(figsize=(8, 6))
        fig5 = sns.kdeplot(data=merged_data, 
                       x="Perpetrator_Sex_encoded", 
                       y="Overall_Unemployment_Rate", 
                       fill=True)
        plt.title("2D KDE Plot of Perpetrator Sex vs Overall Unemployment Rate")
        plt.xlabel("Encoded Perpetrator Sex")
        plt.ylabel("Overall Unemployment Rate")
        st.pyplot(plt.gcf())
    
    elif selection_graph_eda == "Average Homicide Victim Count by State":
        fig6 = px.bar(
            merged_data.groupby('State')[['Victim Count', 'Overall_Unemployment_Rate']].mean().reset_index(),
            x='State',
            y='Victim Count',
            color='Overall_Unemployment_Rate',
            title='Average Homicide Victim Count by State and Unemployment Rate',
            labels={'Victim Count': 'Average Homicide Victim Count', 
                'Overall_Unemployment_Rate': 'Overall Unemployment Rate (%)'}
        )
        st.plotly_chart(fig6)


