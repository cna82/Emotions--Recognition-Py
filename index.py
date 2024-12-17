# Imports Librarires (MNE-Py , Pandas , Sckit-learn)
import pandas as pd
import mne 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 

# بارگذاری داده‌ها 
data = pd.read_csv('eeg_data.csv')

# فرض میشود که داده ها شامل ستون هایی برای سیگنال های ای ای جی  و برچسب های احساسات اند
eeg_signals = data.iloc[:, :-1] # سیگنال‌های EEG 
labels = data.iloc[:, -1] # برچسب‌های احساسات

#  تقسیم داده ها به مجموعه آموزشی و تست 
X_train, X_test, y_train, y_test = train_test_split(eeg_signals, labels, test_size=0.2, random_state=42)

# استخراج ویژگی ها 
from scipy.fft import fft
def extract_features(signals):
    features=[]
    for signal in signals:
        # انجام FFT 
            yf=fft(signal)
            # استخراج ویژگی های فرکانس 
            features.append(np.abs(yf[:len(yf)//2]))
            return np.array(features)
X_train_features=extract_features(X_train.values)
X_test_features=extract_features(X_test.values)  

# آموزش مدل و ارزیابی 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# مدل جنگل تصادفی RandomForestClassifier 
clf = RandomForestClassifier(n_estimators=100, random_state=42) 
clf.fit(X_train, y_train) 

# پیش بینی 
y_pred = clf.predict(X_test)

# ارزیابی دقت
accuracy = accuracy_score(y_test, y_pred) 
print(f'Accuracy: {accuracy*100:.2f}%')
 
# نمایش مارتیکس سردرگمی confusion matrix 
cm = confusion_matrix(y_test, y_pred) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted') 
plt.ylabel('Actual') 
plt.title('Confusion Matrix') 
plt.show()