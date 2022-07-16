import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier 

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

class_list = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
@st.cache()
def prediction(model,feature_col):
  glass_type = model.predict([feature_col])
  if glass_type == 1:
    return 'building windows float processed'.upper()
  elif glass_type == 2:
    return 'building windows non float processed'.upper()
  elif glass_type == 3:
    return 'vehicle windows float processed'.upper()
  elif glass_type == 4:
    return 'vehicle windows non float processed'.upper()
  elif glass_type == 5:
    return 'containers'.upper()
  else:
    return 'headlamp'.upper()

st.title('Glass Type Prdiction Webapp')
st.sidebar.title('Glass Type Prediction Webapp')

if st.sidebar.checkbox('Show Raw Data'):
  st.subheader('Glasstype Dataset')
  st.dataframe(glass_df)

# S1.1: Add a multiselect widget to allow the user to select multiple visualisation.
st.sidebar.subheader('visualisation selector')
lst = st.sidebar.multiselect('Select Charts/Plots:',('Correlation Heatmap','Line Chart','Area Chart','Count Plot','Pie Chart','Box Plot'))

# S1.2: Display Streamlit native line chart and area chart
if 'Line Chart' in lst:
  st.subheader('Line Chart')
  st.line_chart(glass_df)
if 'Area Chart' in lst:
  st.subheader('Area Chart')
  st.area_chart(glass_df)

# S1.3: Display the plots when the user selects them using multiselect widget.
st.set_option('deprecation.showPyplotGlobalUse', False)
if 'Correlation Heatmap' in lst:
  st.subheader('Correlation Heatmap')
  plt.figure(figsize = (14,5))
  sns.heatmap(glass_df.corr(), annot = True)
  st.pyplot()
if 'Count Plot' in lst:
  st.subheader('Count Plot')
  st.countplot(glass_df)
if 'Pie Chart' in lst:
  st.subheader('Pie Chart')
  p = glass_df['GlassType'].value_counts()
  plt.figure(figsize = (5,5))
  plt.pie(p,label = p.index, autopct = '%1.2f%%', startangle = 30, explode = np.linspace(0.6,0.12,6))
  st.pie_plot()

# S1.4: Display box plot using matplotlib module and 'st.pyplot()'
if 'Box Plot' in lst:
  st.subheader('Box Plot')
  column = st.sidebar.selectbox('Select column for Box Plot',('RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','GlassType'))
  sns.boxplot(glass_df[column])
  st.pyplot()

# S2.1: Add 9 slider widgets for accepting user input for 9 features.
st.sidebar.subheader('Select Values')
ri = st.sidebar.slider('Input Ri',float(glass_df['RI'].min()),float(glass_df['RI'].max()))
na = st.sidebar.slider('Input Na',float(glass_df['Na'].min()),float(glass_df['Na'].max()))
mg = st.sidebar.slider('Input Mg',float(glass_df['Mg'].min()),float(glass_df['Mg'].max()))
al = st.sidebar.slider('Input Al',float(glass_df['Al'].min()),float(glass_df['Al'].max()))
si = st.sidebar.slider('Input Si',float(glass_df['Si'].min()),float(glass_df['Si'].max()))
k = st.sidebar.slider('Input K',float(glass_df['K'].min()),float(glass_df['K'].max()))
ca = st.sidebar.slider('Input Ca',float(glass_df['Ca'].min()),float(glass_df['Ca'].max()))
ba = st.sidebar.slider('Input Ba',float(glass_df['Ba'].min()),float(glass_df['Ba'].max()))
fe = st.sidebar.slider('Input Fe',float(glass_df['Fe'].min()),float(glass_df['Fe'].max()))

# S3.1: Add a subheader and multiselect widget.
st.sidebar.subheader('Select Classifier')
c = st.sidebar.selectbox('classifier', ('SVM','RFC','LogR'))

# S4.1: Implement SVM with hyperparameter tuning
from sklearn.metrics import plot_confusion_matrix
if c == 'SVM':
  st.sidebar.subheader('Model HyperParameter')
  cval = st.sidebar.number_input('error rate',1,100,step = 1)
  kval = st.sidebar.radio('kernel',('linear','rbf','poly'))
  gval = st.sidebar.number_input('Gamma',1,100,step = 1)
  if st.sidebar.button('Classify'):
    st.subheader('SVM')
    svc = SVC(C = cval,kernel = kval, gamma = gval).fit(X_train,y_train)
    p = svc.predict(X_test)
    acc = svc.score(X_test,y_test)
    glass_type = prediction(svc,(ri,na,mg,al,si,k,ca,ba,fe))
    st.write('The type of glass predicted is ',glass_type)
    st.write('Accuracy ',acc.round(2))
    plot_confusion_matrix(svc,X_test,y_test)
    st.pyplot()

if c =='RFC':
  st.sidebar.subheader("Model Hyperparameters")
  n_estimators_input = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step = 10)
  max_depth_input = st.sidebar.number_input("Maximum depth of the tree", 1, 100, step = 1)

  # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
  # This 'if' statement must be inside the above 'if' statement. 
  if st.sidebar.button('Classify'):
      st.subheader("Random Forest Classifier")
      rf_clf= RandomForestClassifier(n_estimators = n_estimators_input, max_depth = max_depth_input, n_jobs = -1)
      rf_clf.fit(X_train,y_train)
      accuracy = rf_clf.score(X_test, y_test)
      glass_type = prediction(rf_clf,( ri, na, mg, al, si, k, ca, ba, fe))
      st.write("The Type of glass predicted is:", glass_type)
      st.write("Accuracy", accuracy.round(2))
      plot_confusion_matrix(rf_clf, X_test, y_test)
      st.pyplot()

# S1.1: Implement Logistic Regression with hyperparameter tuning
if c == 'LogR':
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st.sidebar.number_input("C", 1, 100, step = 1)
    max_iter_input = st.sidebar.number_input("Maximum iterations", 10, 1000, step = 10)

    # If the user clicks the 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement.
    if st.sidebar.button('Classify'):
        st.subheader("Logistic Regression")
        log_reg = LogisticRegression(C = c_value, max_iter = max_iter_input)
        log_reg.fit(X_train, y_train)
        accuracy = log_reg.score(X_test, y_test)
        glass_type = prediction(log_reg, (ri, na, mg, al, si, k, ca, ba, fe))
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(log_reg, X_test, y_test)
        st.pyplot()