import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from lime import lime_tabular
import pickle


# Build `wrangle` function
def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)
        
    # replace missing values with 0
    df = df.fillna(0)
    
    # replace rows that indicate ownership with 'Owned' in the 'Owned for' field
    df['Owned for'] = df['Owned for'].replace(['<= 6 months', '> 6 months'],'Owned')
    
    # encode the categorical variables to numeric; Never owned/owned
    def owned(x):
        if x == 'Never owned':
            return int(1)
        else:
            return int(0)
        
    df['Owned for'] = df['Owned for'].apply(owned)

    # for modeling purpose, encode categorical variables to numeric using Ordinal Encoder
    cols = ['Used it for', 'Model Name']
    ord_enc = OrdinalEncoder()
    df[cols] = ord_enc.fit_transform(df[cols])
    
    return df

# Use wrangle function and explore the data
df = wrangle("data/moped.csv")

# selecting features and target data
X = df.drop("Owned for", axis=1)
y = df["Owned for"]

# split data into train and test sets
# 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42)

# the exploratory data analysis showed that the dataset is imbalanced, oversampling is required to balance the dataset

X_train, y_train = SMOTE().fit_resample(X_train, y_train)
X_test, y_test = SMOTE().fit_resample(X_test, y_test)

# standardize data
ss = StandardScaler()
scalled_x_train = ss.fit_transform(X_train)
scalled_x_test = ss.transform(X_test)


# create an instance of the random forest classifier and train on the train set
clf = RandomForestClassifier(n_estimators=45, criterion='entropy', n_jobs=-1).fit(scalled_x_train, y_train)

#  save pickle filee
pickle.dump(clf, open('rf_model.pkl','wb'))

# load model
rf_classif = pickle.load(open('rf_model.pkl','rb'))

# Predict the outcomes on the test set
y_pred = rf_classif.predict(scalled_x_test)

## Dashboard
st.title("Moped Bike Prediction :bar_chart: :bike: ")
st.markdown("Predict Moped owned using reviews")

tab1, tab2, tab3 = st.tabs(["Data :clipboard:", "Global Performance :weight_lifter:", "Local Performance :bicyclist:"])


with tab1:
    st.header("Moped Dataset")
    st.write(df)

with tab2:
    st.header("Confusion Matrix | Feature Importances")
    col1, col2 = st.columns(2)
    with col1:
        conf_mat_fig = plt.figure(figsize=(6,6))
        ax1 = conf_mat_fig.add_subplot(111)
        skplt.metrics.plot_confusion_matrix(y_test, y_pred, ax=ax1, normalize=True)
        st.pyplot(conf_mat_fig, use_container_width=True)

    with col2:
        feat_imp_fig = plt.figure(figsize=(6,6))
        ax1 = feat_imp_fig.add_subplot(111)
        skplt.estimators.plot_feature_importances(rf_classif, feature_names=X.columns, ax=ax1, x_tick_rotation=90)
        st.pyplot(feat_imp_fig, use_container_width=True)

    st.divider()
    st.header("Classification Report")
    st.code(classification_report(y_test, y_pred))     
        
with tab3:
    sliders = []
    col1, col2 = st.columns(2)
    with col1:
        for ingredient in X.columns:
            ing_slider = st.slider(label=ingredient, min_value=float(df[ingredient].min()), max_value=float(df[ingredient].max()))
            sliders.append(ing_slider)

    with col2:
        col1, col2 = st.columns(2, gap="medium")
        
        prediction = rf_classif.predict([sliders])
               
        with col1:
            if prediction[0] == 0:
                st.markdown("### Model Prediction : <strong style='color:tomato;'>Owned</strong>", unsafe_allow_html=True)
            else:
                st.markdown("### Model Prediction : <strong style='color:tomato;'>Never Owned</strong>", unsafe_allow_html=True)
            
            probs = rf_classif.predict_proba([sliders])           
            probability = probs[0][prediction[0]]
        
        with col2:
            st.metric(label="Model Confidence", 
                      value="{:.2f} %".format(probability*100), 
                      delta="{:.2f} %".format((probability-0.5)*100))

        # Rename columns to avoid whitespace and special characters
        df.columns = df.columns.str.replace(' ', '_')   
        df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)


        # Get feature names for LIME explainer
        feature_names = list(df.drop(columns=['Owned_for']).columns.values)


        # Initialize LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(X_train.values, 
                                                      mode="classification", 
                                                      categorical_features=[i for i in range(len(feature_names))], 
                                                      feature_names=feature_names)
        explanation = explainer.explain_instance(np.array(sliders), rf_classif.predict_proba, num_features=len(feature_names), top_labels=3)
        interpretation_fig = explanation.as_pyplot_figure(label=prediction[0])
        st.pyplot(interpretation_fig, use_container_width=True)
        
        
st.text('')
st.text('')
st.markdown('`Code:` [GitHub](https://github.com/yusufokunlola/moped-prediction)')