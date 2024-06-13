 # Price prediction of real estate in St.Petersburg

 ## Data source 

 Data from Yandex.Realty classified https://realty.yandex.ru containing real estate listings for apartments in St. Petersburg and Leningrad Oblast from 2016 till the middle of August 2018. Initial data contains data on sell and rent. 

  ## EDA

  Column information:

  * offer_id: Unique identifier of the real estate offer/offer.
  * first_day_exposition: The date the offer was first posted.
  * last_day_exposition: The date the offer was last posted.
  * last_price: The last price of the property.
  * floor: The floor on which the property is located.
  * open_plan: Flag indicating if the layout is open.
  * rooms: The number of rooms in the property.
  * studio: Flag indicating if the property is a studio. 9. 
  * area: The total area of the property.
  * kitchen_area: The area of the kitchen.
  * living_area: Living area.
  * agent_fee: Agent's commission.
  * renovation: Renovation costs.
  * offer_type: 2 - RENT, 1 - SELL
  * category_type: Category type.
  * unified_address: The unified address of the property.
  * building_id: Unique building identifier.
  * We divided the table into two: the first with all rental information, the second with sales information.

  For prediction model we use only rent data (offer_type: 2) in Saint-Petersburg.

  ```
  rent_df_spb = rent_df[rent_df.unified_address.str.contains('Россия, Санкт-Петербург')].copy()

  ```
  We added feautures like last_price_log, price_per_sq_m, house_price_sqm_median.
  
  We cleaned data from outliers:

  ```
  rent_df_cleaned = rent_df_spb[~((rent_df_spb.price_per_sq_m/rent_df_spb.house_price_sqm_median) > 5)]

  rent_df_cleaned = rent_df_cleaned[rent_df_cleaned.last_price < 1000000]

  rent_df_cleaned = rent_df_cleaned[~((rent_df_cleaned.price_per_sq_m > 3000) & ((rent_df_cleaned.house_price_sqm_median < 1000) | (rent_df_cleaned.house_price_sqm_median == rent_df_cleaned.price_per_sq_m)))]
  
  rent_df_cleaned = rent_df_cleaned[~((rent_df_cleaned.price_per_sq_m < 250) & (rent_df_cleaned.house_price_sqm_median/rent_df_cleaned.price_per_sq_m >= 2))]

  rent_df_cleaned = rent_df_cleaned[~((rent_df_cleaned.price_per_sq_m < 200) & (rent_df_cleaned.price_per_sq_m == rent_df_cleaned.house_price_sqm_median))]
  ```
  The estates with the rent price less than 1000 000 were used, also we delete estate where price per square meter 5 times more house_price_sqm_median. The third line filters out rows where the price per square meter is greater than 3,000 and the house price per square meter is less than 1,000 or equal to the price per square meter. The fourth line filters out rows where the price per square meter is less than 250 and the ratio of the house price per square meter to price per square meter is greater than or equal to 2.The fifth line filters out rows where the price per square meter is less than 200 and the house price per square meter is equal to the price per square meter.


  Initial parameters of rent_cleaned_data and added features like last_price_log, price_per_sq_m, house_price_sqm_median.
  
  ![Alt text](<Снимок экрана 2024-06-12 в 23.10.09.png>)
  
  Also missing values were filled with median area for area factors according to grouped estate by address. Renovation was filled with 0. 

  ![Alt text](<Снимок экрана 2024-06-13 в 23.57.13.png>)
  
  Correlation Heatmap
  
  ![Alt text](<Снимок экрана 2024-06-12 в 23.41.43.png>)

  According to correlation heatmap, for the model we have chosen area, rooms, renovation and open_plan (even though correlation is not so high, it can be iportant for users).
  
  ## Model
  Different types of models were tested like SWM, CatBoost, RandomForestRegressor.

  The best model appeared to be RandomForestRegressor with the following parameters:

  1. bootstrap = True 
   Each tree in the random forest will be trained on a bootstrap sample of the original dataset.
  2. max_depth = 10
   The maximum depth of the trees will be limited to 10, which can help prevent overfitting.
  3. max_features = 2
  This parameter determines the maximum number of features to consider when looking for the best split at each node. In this case, only 2 features will be considered for each split, which can help to reduce the variance of the model.
  4. min_samples_split = 4
  This parameter sets the minimum number of samples required to split an internal node. A node will only be split if it contains at least 4 samples, which can help prevent overfitting by avoiding splitting nodes with too few samples.
  5. n_estimators = 150
  The random forest will contain 150 trees, which can help improve the predictive performance of the model by combining the predictions of multiple trees.

```
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
poly.fit_transform(X)

poly_model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                       ('rf', RandomForestRegressor(
                        bootstrap = True, 
                        max_depth = 10, 
                        max_features = 2, 
                        min_samples_split = 4, 
                        n_estimators = 150))])

poly_model.fit(X_train, Y_train)

print('MAE:', metrics.mean_absolute_error(Y_valid,Y_pred))
print("Training Accuracy = ", poly_model.score(X_train, Y_train))
print("Test Accuracy     = ", poly_model.score(X_valid, Y_valid))
```

Also PolynomialFeatures was used. PolynomialFeatures generates polynomial and interaction features for the input data. A degree of 2 means that it will create polynomial features up to the second degree. This allows for capturing non-linear relationships between the features in the data.

Result: 

Training Accuracy =  0.52
Test Accuracy     =  0.50

The result is not very good but for baseline model it is sufficient.


  ## Virtual environment
    
  Install virtual environment:

  ```pip install virtualenv```

  Activate the virtual environment:

  ```source venv/bin/activate```
  
  Libraries, which are needed to instal using  ```pip install``` are in requirements.txt file:

```
blinker==1.8.2
click==8.1.7
flask==3.0.3
importlib-metadata==7.1.0
itsdangerous==2.2.0
jinja2==3.1.4
joblib==1.4.2
MarkupSafe==2.1.5
numpy==1.24.4
scikit-learn==1.2.2
scipy==1.10.1
threadpoolctl==3.5.0
werkzeug==3.0.3
zipp==3.19.2
```
It is necessary to install flask, joblib, scikit-leran, numpy.
  
To run the code you need to run code in app.py document,using in Treminal

 ```python3 app.py ```
  
Thi file contains the following code:

```
from flask import Flask, request 
import joblib 
import numpy 

MODEL_PATH = 'mlmodels/model.pkl'
SCALER_X_PATH = 'mlmodels/scaler_x.pkl'
SCALER_Y_PATH = 'mlmodels/scaler_y.pkl'


app = Flask(__name__)
model = joblib.load(MODEL_PATH)
sc_x = joblib.load(SCALER_X_PATH)
sc_y = joblib.load(SCALER_Y_PATH)

@app.route("/predict_price",methods = ['GET']) 
def predict():
    args = request.args
    open_plan = args.get('open_plan',default = -1,type = int)
    rooms = args.get('rooms',default = -1,type = int)
    area = args.get('area',default = -1,type = float)
    renovation = args.get('renovation',default = -1,type = float)

    #response = 'open_plna:{}, rooms:{}, area: {}, renovation: {}'.format(open_plan, rooms,area,renovation)
    x = numpy.array([open_plan, rooms, area, renovation]).reshape(1,-1)
    x = sc_x.transform(x)
    result = model.predict(x)
    result = sc_y.inverse_transform(result.reshape(1,-1))
    return str(result[0][0])


if __name__ == "__main__": 
    app.run(debug=True,host ='0.0.0.0',port=7778)

```
 ### Explanation of the code:
 This Flask application loads ML model (model.pkl) and scalar objects (scaler_x.pkl, scaler_y.pkl) for feature scaling. It implements a /predict_price endpoint which takes query parameters (open_plan, rooms, area, renovation) from the URL and returns the predicted price based on these input values using the loaded model.

1. Load Model and Scalars: Load the trained ML model and scalar objects from the specified paths (MODEL_PATH) using joblib.load.

3. Flask App: Create a Flask app named app.

4. Predict Function: Define a route /predict_price to handle requests for price prediction. Extract the input variables open_plan, rooms, area, and renovation from the request arguments. Reshape the input into a numpy array, apply feature scaling, run the prediction using the loaded model, and then perform inverse scaling on the result to get the predicted price.

5. Response: Return the predicted price as a string in the response.


  ## Dockerfile
  ```
    FROM python:3.9-slim
    MAINTAINER Polina Okolo-Kulak
    RUN apt-get update -y
    COPY . /opt/gsom_predictor
    WORKDIR /opt/gsom_predictor
    RUN apt install -y python3-pip 
    RUN pip3 install -r requirements.txt
    CMD python3 app.py
```
1. Docker image based on the Python 3.9-slim image
2. Updates the package lists in the image using apt-get
3. Copies the current directory into a directory called "gsom_predictor" in the image
4. Sets the working directory to "gsom_predictor"
5. Installs Python 3 and pip using apt
6. Installs the Python packages listed in the requirements.txt file using pip
7. Runs the command "python3 app.py" as the default command when the container is started

 ### In conclusion 

This Dockerfile is setting up an environment for running a Python application called app.py which presumably has some functionality related to the gsom_predictor project.
  
## Port in remote VM

Chosen port for running the code in remore machine: 7778 
To open this port for web traffic in terminal on remote machine use:

```
sudo ufw allow 7778
```
Verify that the port has been successfully opened by running the following command:

```
sudo ufw status
```

## Run app using Docker

Port: 7778 

To run this Docker container we need try the folowwing command:

```
docker run --network host -d okolokulak/gsom_e2e24:v.0.4
```
If you want to stop it, you can provide the followin command 

```
docker stop NAME
```
Example of working web service in Postman:

![Alt text](<Снимок экрана 2024-06-12 в 23.22.40.png>)