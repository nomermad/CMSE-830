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
                            "Linear Regression for Unemployment Rates based on Highest Degree",
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

    elif selection_graph_eda == "Linear Regression for Unemployment Rates based on Highest Degree":
        x = merged_data['Overall_Unemployment_Rate'].values.reshape(-1, 1)  #using numpy reshape to reshape the data to a 2D array, wouldn't run without this
        high_school = merged_data['High_School'] 
        associates = merged_data['Associates_Degree'] 
        professional = merged_data['Professional_Degree']  


        high_school_linreg = LinearRegression()
        high_school_linreg.fit(x, high_school)

        associates_linreg = LinearRegression()
        associates_linreg.fit(x, associates)

        professional_linreg = LinearRegression()
        professional_linreg.fit(x, professional)


        pred_high_school = high_school_linreg.predict(x)
        pred_associates = associates_linreg.predict(x)
        pred_professional = professional_linreg.predict(x)

        rsquared_high_school = high_school_linreg.score(x, high_school)
        rsquared_associates = associates_linreg.score(x, associates)
        rsquared_professional = professional_linreg.score(x, professional)
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(x, high_school, label='High School Degree Data')
        ax.plot(x, pred_high_school, label='High School Fit', linestyle='--')

        ax.scatter(x, associates, label='Associates Degree Data')
        ax.plot(x, pred_associates, label='Associates Degree Fit', linestyle='--')

        ax.scatter(x, professional, label='Professional Degree Data')
        ax.plot(x, pred_professional, label='Professional Degree Fit', linestyle='--')

        ax.set_title('Multivariate Linear Regression')
        ax.set_xlabel('Overall Unemployment Rate')
        ax.set_ylabel('Education Levels')
        ax.legend()
        st.pyplot(fig)
        st.write(f'R² for High School: {rsquared_high_school:.2f}')
        st.write(f'R² for Associates Degree: {rsquared_associates:.2f}')
        st.write(f'R² for Professional Degree: {rsquared_professional:.2f}')
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


