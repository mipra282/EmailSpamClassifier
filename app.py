from flask import Flask,render_template,request,url_for
from sklearn.feature_extraction.text import CountVectorizer
import pickle

app = Flask(__name__)

msg_list = []

@app.route('/',methods=['GET'])
def home():
	
	return render_template('homepage.html')


@app.route('/classify',methods=['POST'])
def predict():
	if request.method == 'POST':

		sent = request.form.get('msg')

		with open('nbspam.pkl', 'rb') as f:
			clf = pickle.load(f)

		with open('vect.pkl', 'rb') as f:
			vect = pickle.load(f)

	
		cvobj = vect.transform([sent])
		prediction = clf.predict(cvobj.toarray())

	

		msg_list.append(sent)

		

		if prediction[0] != 0:
			classed = 'Spam'
			boot = "badge badge-danger"

		else:
			classed = "Not Spam"
			boot = "badge badge-success"

		return render_template('homepage.html',classed=classed,boot=boot,msg_list=msg_list)


    



if __name__ == "__main__":
	app.run(port=5000,debug=True)