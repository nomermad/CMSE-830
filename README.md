# CMSE-830

  The streamlit app contains a lot of useful information and includes a lot of interactive features to do so. To start, the app is composed of four tabs on the navigation bar, which has data processing, exploratory data analysis, the machine learning models, and visualization. Under the data processing tab, you can view the homicide data, the unemployment data, and the crime data. There is an interactive drop down menu, where you can choose what table you want to see. Then, the missing values of both tables were put into a heatmap, which is interactive as well and you can use the drop down menu to choose what table you want to see. To move to the exploratory data analysis page, you can select it from the drop down menu. The first interactive feature on this page is choosing to look at specific features in a table and comparing those features. Then there is an option to click a checkbox and see relevant statistics for specific comlumns. The mean value was found for relevant columns, and you can click the checkbox to display which mean values you would like to look at. There then is a dropdown button to view different correlation matrix's with different features. More specifcally, one has unemployment features and homicide features, and the other one has crime features and unemployment features. Moving on, the last interactive element allows you to click on specific columns to view their z-scores in a table.  
    Moreover, you can then move into the machine learning tab to view machine learning models, such as linear regression, knn, and rfr. There is the option to view statistics from the different models or view the graphs using the models. The models were trained and tested on specific features. 
  Finally, you can see the visualizations by clicking on the visualization tab. 12 graphs were created, and you can click the drop down button to choose which graph you want to view.

Python Requirement: 3.12.4
Pandas Requirement: 2.2.2
Numpy Requirement: 1.26.4


The following packages are important and needed for the project: numpy, pandas, matplotlib, seaborn, sklearn.model_selection, sklearn.linear_model, plotly, and sklearn.preprocessing. 

Link to my streamlit app: https://nomermad-cmse-830-app-6ymnml.streamlit.app/
