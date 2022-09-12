#Importing all the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from flask import Flask

#from Cluster_Main import * 
from flask import flash, request, redirect, render_template,url_for,sessions
import pickle
import ast
import warnings
warnings.filterwarnings('ignore')
#Read the input data
df = pd.read_excel("BMI_Data.xlsx")
#clean the data
def treat_outliers(field):
    q1 = field.quantile(0.25)
    q2 = field.quantile(0.50)
    q3 = field.quantile(0.75)
    iqr = q3 - q1
    lower_boundary = q1 - (1.5* iqr)
    upper_boundary = q3 + (1.5* iqr)
    new_col = pd.Series(np.where(field < lower_boundary, q2, field))
    new_col2 = pd.Series(np.where(new_col > upper_boundary, q2, new_col))
    return new_col2
#Apply treat_outliers function on the fields in the data
df["Weight in Pounds"]=treat_outliers(df["Weight in Pounds"])
df["BMI"]=treat_outliers(df["BMI"])
df["Cholesterol"]=treat_outliers(df["Cholesterol"])
x_norm = MinMaxScaler()
y_norm =  MinMaxScaler()
#Seperating independent variables and target variable
Xx = df.drop(["Cholesterol", "Date"], axis=1)
yy = df["Cholesterol"]
#Create model to transform data
xtrans = x_norm.fit(Xx)
#Scale date with the model created
xtrans_data = xtrans.transform(Xx)
class y_transformer:
    def __init__(self, yv):
        self.yv =yv
        self.mins = min(self.yv)
        self.maxs = max(self.yv)
    def transform(self):
        return self.yv.apply(lambda x: ((x-self.mins)/(self.maxs - self.mins)))
    def single_inverse_transform(self, single):
        return self.mins + ((self.maxs - self.mins)*single)
    def df_inverse_transform(self, col):
        return col.apply(lambda single: self.mins + ((self.maxs - self.mins)*single))
ytrans = y_transformer(yy)
ytrans_data = ytrans.transform()
#Scaled data is in array data type. So change it into dataframe datatype
scaled_df = pd.DataFrame(xtrans_data, columns=["Weight", "BMI"])

#Splitting the data into train, test data for independent and target variables
x_train, x_test, y_train, y_test = train_test_split(scaled_df,ytrans_data,test_size = 0.25, random_state=128)
#Create instance for the LinearRegression
regression = LinearRegression()

regression.fit(x_train, y_train)

def get_cholestrol_for_single(array_in):
    #weight = ast.literal_eval(input("Enter the Weight in pounds:"))
    #bmi = ast.literal_eval(input("Enter the BMI:"))
    inputs = xtrans.transform(array_in)
    new_y = regression.predict(inputs)
    new_y = ytrans.single_inverse_transform(new_y[0])
    return new_y

def get_cholestrol_for_csv(inputs):
    # loca = input("Please enter the full file path of csv file [Ex: C:\docs\bmi.csv]:")
    # if loca.strip('"')[-3:]=='csv':
    #     try:
    #         inputs = pd.read_csv(loca.strip('"'))
    #     except FileNotFoundError as e:
    #         return e
    # elif loca.strip('"')[-4:]=='xlsx':
    #     try:
    #         inputs = pd.read_excel(loca.strip('"'))
    #     except FileNotFoundError as e:
    #         return e
    # else:
    #     return "Please give valid file path with file name"
    inputs_t = xtrans.transform(inputs)
    new_y = regression.predict(inputs_t)
    new_ans = pd.concat([inputs, ytrans.df_inverse_transform(pd.DataFrame(new_y, columns = ["Cholesterol"]))], axis = 1)
    return new_ans
    

app = Flask(__name__)
app.secret_key = "secret key"

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(['csv','xlsx'])

#pd.set_option('display.max_colwidth', -1)


def ValuePredictor(to_predict_list):
    to_predict_list = pd.DataFrame(to_predict_list, columns = ["Weight", "BMI"])
    
    result = regression.predict(to_predict_list)
    return result[0]

@app.route('/', methods = ['GET', 'POST'])
def resultss():
    
    
    return render_template("test_divs.html")

@app.route('/results', methods = ['GET', 'POST'])
def results():
    print(request)
    weight = ast.literal_eval(request.args.get('weight'))
    bmi = ast.literal_eval(request.args.get('bmi'))
    print(weight)
    print(bmi)
    result_single = round( get_cholestrol_for_single(np.array([[weight, bmi]])), 4)
    print(result_single)
    texts = f"The Cholesterol level for weight {weight} pounds and {bmi} bmi index is:"
    return {"ans":str(result_single), "texts": texts}#render_template("index.html", result_one = result_single)

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/csvout', methods=['GET','POST'])
def upload_file():
    print("*******************************************", request.args.get('fname'))
    file =request.args.get('fname')
    
    global csv_inputs
    if file.strip('"')[-3:]=='csv':
        try:
            csv_inputs = pd.read_csv(file.strip('"'))
        except FileNotFoundError as e:
            return e
    elif file.strip('"')[-4:]=='xlsx':
        try:
            csv_inputs = pd.read_excel(file.strip('"'))
        except FileNotFoundError as e:
            return e
  
    else:
        return "Please give valid file path with file name"
    
    flash('File successfully uploaded')

    answer = get_cholestrol_for_csv(csv_inputs)
    return render_template('csvresult.html', outp = answer)# tuple(ast.literal_eval(answer.to_json(orient='records')))


if __name__ == "__main__":
    app.run() 
    
    



