import os,pickle,pandas as pd,matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
B=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(B,"data/resume_dataset.csv"); M=os.path.join(B,"model"); os.makedirs(M,exist_ok=True)
df=pd.read_csv(D).dropna();X=df.resume.astype(str);y=df.role.astype(str)
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=Pipeline([("tfidf",TfidfVectorizer(lowercase=True,stop_words="english",ngram_range=(1,2),sublinear_tf=True)),("classifier",RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced"))])
model.fit(Xt,yt);p=model.predict(Xv);acc=accuracy_score(yv,p)
print(f"Dataset: {len(df)} samples | Roles: {df.role.nunique()} | Test accuracy: {acc*100:.2f}%")
print(classification_report(yv,p,zero_division=0))
pickle.dump(model,open(os.path.join(M,"resume_model.pkl"),"wb"));open(os.path.join(M,"model_report.txt"),"w",encoding="utf8").write(f"Dataset size: {len(df)}\nNumber of job roles: {df.role.nunique()}\nTest accuracy: {acc*100:.2f}%\n\n"+classification_report(yv,p,zero_division=0))
labels=sorted(y.unique());fig,ax=plt.subplots(figsize=(12,10));ConfusionMatrixDisplay(confusion_matrix(yv,p,labels=labels),display_labels=labels).plot(ax=ax,xticks_rotation=90,colorbar=False);plt.tight_layout();fig.savefig(os.path.join(M,"confusion_matrix.png"),dpi=160);plt.close(fig)
print("Model saved to model/resume_model.pkl")
