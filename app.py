

import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import altair as alt
import streamlit.components.v1 as components
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric

# Custome Component Fxn
import sweetviz as sv 
import codecs

def st_display_sweetviz(report_html,width=1000,height=500):
	report_file = codecs.open(report_html,'r')
	page = report_file.read()
	components.html(page,width=width,height=height,scrolling=True)


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Solar Energy Dashboard", page_icon=":bar_chart:", layout="wide")


st.title("Classification Web App - Solar Power Generation")
st.subheader("Creator : Debmalya Ray")

st.sidebar.title("Categorical Classification")

@st.cache(persist=True)

def load_data():
    data = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')
    data["month"] = pd.to_datetime(data["DATE_TIME"]).dt.month
    data["day"] = pd.to_datetime(data["DATE_TIME"]).dt.day
    data["year"] = pd.to_datetime(data["DATE_TIME"]).dt.year

    return data

def load_data2():
    data2 = pd.read_csv('Plant_1_Generation_Data.csv')
    data2["month"] = pd.to_datetime(data2["DATE_TIME"]).dt.month
    data2["day"] = pd.to_datetime(data2["DATE_TIME"]).dt.day
    data2["year"] = pd.to_datetime(data2["DATE_TIME"]).dt.year
    return data2

df = load_data()
df2 = load_data2()

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2016/02/15/11/38/background-1201003_960_720.jpg");
             background-attachment: fixed;
             background-size: cover;
			 color: Red;
			 font-family:sans-serif;
         }}

         </style>
         """,
         unsafe_allow_html=True
     )

	 

add_bg_from_url() 

st.subheader("Weather Sensor Data-1")
st.write(df.head(2))
st.subheader("Generation Data-1")
st.write(df2.head(2))


def main():

	"""Semi Automated ML App with Streamlit """

	activities = ["EDA","Plots", "Plots2", "Sweetviz"]

	choice = st.sidebar.selectbox("Select Activities",activities)

	if choice == 'EDA':
		st.subheader("Exploratory Data Analysis")

		df = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')
		if df is not None:
			if st.checkbox("Show Shape"):
				st.write(df.shape)

			if st.checkbox("Show Columns"):
				all_columns = df.columns.to_list()
				st.write(all_columns)

			if st.checkbox("Summary"):
				st.write(df.describe())

			if st.checkbox("Show Selected Columns"):
				selected_columns = st.multiselect("Select Columns",all_columns)
				new_df = df[selected_columns]
				st.dataframe(new_df)

			if st.checkbox("Show Value Counts"):
				st.write(df.iloc[:,-1].value_counts())

			if st.checkbox("Correlation Plot(Seaborn)"):
				st.write(sns.heatmap(df.corr(),annot=True))
				st.pyplot()

			if st.checkbox("Pair Plot(Seaborn)"):
				st.write(sns.pairplot(df, hue="IRRADIATION"))
				st.pyplot()

	elif choice == 'Plots':
		st.subheader("Data Visualization - Plots")

		df3 = pd.read_csv('Plant_1_Generation_Data.csv')
		if df3 is not None:
		
			# Customizable Plot
			all_columns_names = df3.columns.tolist()
			type_of_plot = st.selectbox("Select Type of Plot",["area","line","kde"])
			selected_columns_names = st.multiselect("Select Columns To Plot(1 column)",all_columns_names)
			if st.button("Generate Plot"):
				st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))
				if type_of_plot == 'area':
					cust_data = df3[selected_columns_names]
					st.area_chart(cust_data)
				elif type_of_plot == 'line':
					cust_data = df3[selected_columns_names]
					st.line_chart(cust_data)
				# Custom Plot 
				elif type_of_plot:
					cust_plot= df3[selected_columns_names].plot(kind=type_of_plot)
					st.write(cust_plot)
					st.pyplot()


	elif choice == 'Plots2':
		
		dataframe = pd.read_csv('Plant_2_Generation_Data.csv')

		dataframe["Year"] = pd.to_datetime(dataframe["DATE_TIME"], format="%Y-%m-%d %H:%M:%S").dt.year
		dataframe["Month"] = pd.to_datetime(dataframe["DATE_TIME"], format="%Y-%m-%d %H:%M:%S").dt.month
		dataframe["Hour"] = pd.to_datetime(dataframe["DATE_TIME"], format="%Y-%m-%d %H:%M:%S").dt.hour
		data_by_month = dataframe.groupby(by=["Month"]).sum()[["TOTAL_YIELD", "DAILY_YIELD", "AC_POWER", "DC_POWER"]]


		hourly_frac = dataframe.groupby(['Hour']).mean()/np.sum(dataframe.groupby(['Hour']).mean())

		#st.write(hourly_frac.columns)
		## st.write(sns.lineplot(data=hourly_frac["TOTAL_YIELD"]))
		### st.write(sns.lineplot(data=hourly_frac, x="Month", y="DAILY_YIELD"))
		st.subheader("Plots 1 - Line Plot")
		st.write(sns.lineplot(data=hourly_frac))
		st.pyplot()

		st.subheader("Plots 2 - KDE Plot")
		st.write(sns.kdeplot(data=hourly_frac, x="TOTAL_YIELD", hue="Month", multiple="stack"))
		st.pyplot()

		st.subheader("Plots 3 - Cluster MAP")
		## st.write(sns.residplot(data=hourly_frac, x="TOTAL_YIELD", y="DC_POWER"))
		st.write(sns.clustermap(hourly_frac))
		st.pyplot()

		st.subheader("Plots 4 - JOINT PLOT")

		st.write(sns.jointplot(
    	data=hourly_frac, x="DAILY_YIELD", y="TOTAL_YIELD",
    	marker="+", s=100, marginal_kws=dict(bins=25, fill=False),
		))

		st.pyplot()
		
		
	elif choice == "Sweetviz":
		st.subheader("Automated EDA with Sweetviz")
		st_display_sweetviz("generation_data.html")


if __name__ == '__main__' :
    main()

