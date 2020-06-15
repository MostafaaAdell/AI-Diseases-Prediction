# import libraries
import sys
from PyQt5 import QtWidgets
import pandas as pd
from  PyQt5.QtGui import QPixmap

# Read Data From Exal
cancer = pd.read_csv("C:/Users/user/Desktop/cancer.csv")
diabetes = pd.read_excel("C:/Users/user/Desktop/diabetes.xlsx")

# Split the data to independed variable and depended 
X = cancer.iloc[:, :-1].values
y = cancer.iloc[:, 8].values
Z = diabetes.iloc[:, :-1].values
w = diabetes.iloc[:, 8].values

# Encoding Outcome Catigorical Data To Normal Data
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

# Scaling Data 
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
stdz=StandardScaler()
X=std.fit_transform(X)
Z=stdz.fit_transform(Z)

#  Split the Data into the Training and Test 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 0 )
Z_train,Z_test,w_train,w_test = train_test_split(Z,w,test_size=0.3,random_state = 0 )


#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
from sklearn.ensemble import RandomForestClassifier
Ran_C = RandomForestClassifier(n_estimators=10,random_state=20)
Ran_D = RandomForestClassifier(n_estimators=10,random_state=20)
Ran_C.fit(X_train,y_train)
Ran_D.fit(Z_train,w_train)

"""#Predicted Value
y_pred = Ran_C.predict(X_test)
w_pred = Ran_D.predict(Z_test)

#Accuracy of Predication
from sklearn.metrics import scorer
accuracy = scorer.accuracy_score(y_test,y_pred)
accuracyz = scorer.accuracy_score(w_test,w_pred)

#confusion_matrix
from sklearn.metrics import confusion_matrix
cm_Rforest = confusion_matrix(y_test, y_pred)
cm_Rforestz = confusion_matrix(w_test, w_pred)
"""


class Diabetes(QtWidgets.QWidget):

        def __init__(self):
            super().__init__()
            
            self.init_ui()
            
        def init_ui(self):
            self.setGeometry(200, 200, 700, 600)
            self.lback=QtWidgets.QLabel(self)
            self.back=QPixmap('C:/Users/user/Desktop/diabetes.jpg')
            self.lback.setPixmap(self.back)
            
            self.label1 = QtWidgets.QLabel(self)
            self.label1.setText('Pregnancies')
            self.label1.move(20, 20)
            self.text1 = QtWidgets.QLineEdit(self)
            self.text1.move(125, 15)
            self.text1.resize(200, 25)
            
            self.label2 = QtWidgets.QLabel(self)
            self.label2.setText('Glucose')
            self.label2.move(20, 80)
            self.text2 = QtWidgets.QLineEdit(self)
            self.text2.move(125, 75)
            self.text2.resize(200, 25)
            
            self.label3 = QtWidgets.QLabel(self)
            self.label3.setText('BloodPressure')
            self.label3.move(20, 140)
            self.text3 = QtWidgets.QLineEdit(self)
            self.text3.move(125, 135)
            self.text3.resize(200, 25)
            
            self.label4 = QtWidgets.QLabel(self)
            self.label4.setText('Skin Thickness')
            self.label4.move(20, 200)
            self.text4 = QtWidgets.QLineEdit(self)
            self.text4.move(125, 195)
            self.text4.resize(200, 25)
            
            self.label5 = QtWidgets.QLabel(self)
            self.label5.setText('insulin')
            self.label5.move(20, 260)
            self.text5 = QtWidgets.QLineEdit(self)
            self.text5.move(125, 255)
            self.text5.resize(200, 25)
            
            self.label6 = QtWidgets.QLabel(self)
            self.label6.setText('BMI')
            self.label6.move(20, 320)
            self.text6 = QtWidgets.QLineEdit(self)
            self.text6.move(125, 315)
            self.text6.resize(200, 25)
            
            self.label7 = QtWidgets.QLabel(self)
            self.label7.setText('DiabetesPeding')
            self.label7.move(20, 380)
            self.text7 = QtWidgets.QLineEdit(self)
            self.text7.move(125, 375)
            self.text7.resize(200, 25)
            
            self.label8 = QtWidgets.QLabel(self)
            self.label8.setText('Age')
            self.label8.move(20, 440)
            self.text8 = QtWidgets.QLineEdit(self)
            self.text8.move(125, 435)
            self.text8.resize(200, 25)
            
            
            
            self.label9 = QtWidgets.QLabel(self)
            self.label9.setText('Diagnosis')
            self.label9.move(20, 550)
            self.text9 = QtWidgets.QLineEdit(self)
            self.text9.move(125, 545)
            self.text9.resize(200, 25)
            
            
            self.Diagnosis_btn = QtWidgets.QPushButton('predict', self)
            self.Diagnosis_btn.move(125, 500)
            self.Diagnosis_btn.resize(200, 25)
            
            
           
            self.setWindowTitle('Diabetes')
    
            self.Diagnosis_btn.clicked.connect(self.Predict_Diabetes)
    
            self.show()
            self.exec()
        def Predict_Diabetes(self):
            Z1 = self.text1.text()
            Z2 = self.text2.text()
            Z3 = self.text3.text()
            Z4 = self.text4.text()
            Z5 = self.text5.text()
            Z6 = self.text6.text()
            Z7 = self.text7.text()
            Z8 = self.text8.text()
            p=[[Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8]]
            p=stdz.transform(p)     
            w_pred = Ran_D.predict(p)
            if str(w_pred) == "[0]":
                self.text9.setText( "Negative")
            elif str(w_pred) == "[1]":
                self.text9.setText( "Postive")




class Cancer(QtWidgets.QWidget):

        def __init__(self):
            super().__init__()
            
            self.init_ui()
            
        def init_ui(self):
            self.setGeometry(200, 200, 600, 600)
            self.lback=QtWidgets.QLabel(self)
            self.back=QPixmap('C:/Users/user/Desktop/cancer.jpg')
            self.lback.setPixmap(self.back)
            
            self.label1 = QtWidgets.QLabel(self)
            self.label1.setText('QPS')
            self.label1.move(20, 20)
            self.text1 = QtWidgets.QLineEdit(self)
            self.text1.move(125, 15)
            self.text1.resize(200, 25)
            
            self.label2 = QtWidgets.QLabel(self)
            self.label2.setText('QPI')
            self.label2.move(20, 80)
            self.text2 = QtWidgets.QLineEdit(self)
            self.text2.move(125, 75)
            self.text2.resize(200, 25)
            
            self.label3 = QtWidgets.QLabel(self)
            self.label3.setText('Proliferation')
            self.label3.move(20, 140)
            self.text3 = QtWidgets.QLineEdit(self)
            self.text3.move(125, 135)
            self.text3.resize(200, 25)
            
            self.label4 = QtWidgets.QLabel(self)
            self.label4.setText('Secretions')
            self.label4.move(20, 200)
            self.text4 = QtWidgets.QLineEdit(self)
            self.text4.move(125, 195)
            self.text4.resize(200, 25)
            
            self.label5 = QtWidgets.QLabel(self)
            self.label5.setText('Estrogen')
            self.label5.move(20, 260)
            self.text5 = QtWidgets.QLineEdit(self)
            self.text5.move(125, 255)
            self.text5.resize(200, 25)
            
            self.label6 = QtWidgets.QLabel(self)
            self.label6.setText('Progesterone')
            self.label6.move(20, 320)
            self.text6 = QtWidgets.QLineEdit(self)
            self.text6.move(125, 315)
            self.text6.resize(200, 25)
            
            self.label7 = QtWidgets.QLabel(self)
            self.label7.setText('Protein receptors')
            self.label7.move(20, 380)
            self.text7 = QtWidgets.QLineEdit(self)
            self.text7.move(125, 375)
            self.text7.resize(200, 25)
            
            self.label8 = QtWidgets.QLabel(self)
            self.label8.setText('Age')
            self.label8.move(20, 440)
            self.text8 = QtWidgets.QLineEdit(self)
            self.text8.move(125, 435)
            self.text8.resize(200, 25)
            
            self.label9 = QtWidgets.QLabel(self)
            self.label9.setText('Diagnosis')
            self.label9.move(20, 550)
            self.text9 = QtWidgets.QLineEdit(self)
            self.text9.move(125, 545)
            self.text9.resize(200, 25)
            
            self.Predict_btn = QtWidgets.QPushButton('predict', self)
            self.Predict_btn.move(125, 500)
            self.Predict_btn.resize(200, 25)
            
           
            self.setWindowTitle('Cancer')
    
            self.Predict_btn.clicked.connect(self.Predict_Cancer)
    
            self.show()
            self.exec()
        def Predict_Cancer(self):
            X1 = self.text1.text()
            X2 = self.text2.text()
            X3 = self.text3.text()
            X4 = self.text4.text()
            X5 = self.text5.text()
            X6 = self.text6.text()
            X7 = self.text7.text()
            X8 = self.text8.text()
            p=[[X1,X2,X3,X4,X5,X6,X7,X8]]
            p=std.transform(p)
            y_pred = Ran_C.predict(p)
            
            if str(y_pred) == "[0]":
                self.text9.setText( "Benign")
            elif str(y_pred) == "[1]":
                self.text9.setText( "Malignant")


def Main_Window():
    app = QtWidgets.QApplication(sys.argv)
    w = QtWidgets.QWidget()
    lback=QtWidgets.QLabel(w)
    back=QPixmap('C:/Users/user/Desktop/main.jpg')
    lback.setPixmap(back)
    cancer_btn = QtWidgets.QPushButton(w)
    cancer_btn.setText('Cancer')
    cancer_btn.setStyleSheet("background-color: red")
    diabetes_btn = QtWidgets.QPushButton(w)
    diabetes_btn.setText('Diabetes')
    diabetes_btn.setStyleSheet("background-color: blue")
    w.setWindowTitle('Predicting_Diseases')
    
    cancer_btn.move(20, 250)
    diabetes_btn.move(315,250)
    w.setGeometry(400, 400, 425, 282)
    cancer_btn.clicked.connect(Cancer)
    diabetes_btn.clicked.connect(Diabetes)
    w.show()
    sys.exit(app.exec_())


Main_Window()


