from flask import Flask, request 

app = Flask(__name__)


@app.route("/predict_price",methods = ['GET']) 
def predict():
    args = request.args
    rooms = args.get('rooms',default = -1,type = int)
    area = args.get('area',default = -1,type = float)
    renovation = args.get('renovation',default = -1,type = float)

    response = 'rooms:{}, area: {}, renovation: {}'.format(rooms,area,renovation)
    
    return response 
if __name__ == "__main__": 
    app.run(debug=True,host ='0.0.0.0',port=7778)