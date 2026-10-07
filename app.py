import os,pickle,subprocess,sys,re
from pathlib import Path
import pandas as pd,streamlit as st
from utils.pdf_extractor import extract_text_from_pdf
from utils.skill_extractor import extract_skills,group_skills
B=Path(__file__).resolve().parent; MP=B/"model/resume_model.pkl"; DP=B/"data/resume_dataset.csv"; RP=B/"model/model_report.txt"; CP=B/"model/confusion_matrix.png"
st.set_page_config(page_title="AI Resume Screening",page_icon="🤖",layout="wide")
st.markdown("<style>.block-container{padding-top:2rem}.hero{padding:2rem;border-radius:20px;background:linear-gradient(135deg,#111827,#374151);color:white;margin-bottom:1.5rem}.hero h1{font-size:2.5rem}.card{padding:1.2rem;border:1px solid #555;border-radius:16px}</style>",unsafe_allow_html=True)
@st.cache_resource
def load_model():
 if not MP.exists(): subprocess.run([sys.executable,str(B/"train_model.py")],check=True)
 return pickle.load(open(MP,"rb"))
@st.cache_data
def data(): return pd.read_csv(DP)
model=load_model();df=data()
st.markdown("<div class='hero'><h1>🤖 AI Resume Screening & Job Recommendation</h1><p>NLP + TF-IDF + Random Forest resume analysis dashboard</p></div>",unsafe_allow_html=True)
with st.sidebar:
 st.header("📌 Project");st.write("**Fundamentals of Artificial Intelligence & Machine Learning**");st.divider();st.write("Training samples:",len(df));st.write("Job categories:",df.role.nunique());st.caption("Academic demonstration only; not a hiring decision.")
t1,t2,t3=st.tabs(["🔍 Resume Analyzer","📊 Model Dashboard","ℹ️ About"])
with t1:
 a,b=st.columns([1.15,1]);
 with a:
  st.subheader("📄 Resume Input");mode=st.radio("Input method",["Upload PDF","Paste Text"],horizontal=True);txt=""
  if mode=="Upload PDF":
   up=st.file_uploader("Upload resume PDF",type=["pdf"]);
   if up:
    try: txt=extract_text_from_pdf(up); st.success("PDF text extracted.") if txt else st.warning("No selectable text found.");
    except Exception as e: st.error(str(e))
  else: txt=st.text_area("Paste resume text",height=300,placeholder="Python, Machine Learning, SQL, Pandas, NumPy...")
  go=st.button("🚀 Analyze Resume",type="primary",use_container_width=True)
 with b:
  st.subheader("🎯 Output");st.markdown("<div class='card'><b>Top job roles</b><br>Model-ranked career categories.<br><br><b>Skill extraction</b><br>Technical skills detected from resume text.<br><br><b>Model dashboard</b><br>Accuracy, report and confusion matrix.</div>",unsafe_allow_html=True)
 if go:
  if not txt.strip(): st.warning("Please upload a PDF or paste resume text.")
  else:
   txt=re.sub(r"\s+"," ",txt).strip(); probs=model.predict_proba([txt])[0]; ranked=sorted(zip(model.classes_,probs),key=lambda x:x[1],reverse=True)[:5]; skills=extract_skills(txt);c1,c2,c3=st.columns(3);c1.metric("Top Role",ranked[0][0]);c2.metric("Top Match",f"{ranked[0][1]*100:.1f}%");c3.metric("Skills",len(skills));st.markdown("### 📊 Top Job Matches");st.bar_chart(pd.DataFrame({"Match (%)":[p*100 for _,p in ranked]},index=[r for r,_ in ranked]));st.markdown("### 🛠️ Detected Skills");
   if skills:
    for g,ss in group_skills(skills).items(): st.write(f"**{g}:** "+", ".join(ss))
   else: st.info("No predefined skills detected.")
   st.markdown("### 💼 Recommended Roles");[st.write(f"**{i}. {r}** — {p*100:.1f}%") for i,(r,p) in enumerate(ranked[:3],1)];st.caption("Scores are classifier probability estimates for this educational dataset, not real hiring probabilities.")
with t2:
 st.subheader("📈 Model Performance")
 if RP.exists():
  report=RP.read_text(encoding="utf8");line=next((x for x in report.splitlines() if x.startswith("Test accuracy:")),"Test accuracy: N/A");st.metric("Test Accuracy",line.split(":",1)[1].strip());
  with st.expander("Classification report"): st.code(report)
 if CP.exists(): st.image(str(CP),caption="Confusion Matrix",use_container_width=True)
 st.subheader("🗂️ Dataset Overview");st.bar_chart(df.role.value_counts());st.dataframe(df.role.value_counts().rename_axis("Role").to_frame("Samples"),use_container_width=True)
with t3:
 st.subheader("📚 AIML Concepts");st.markdown("- Natural Language Processing (NLP)\n- TF-IDF feature extraction\n- Supervised classification\n- Random Forest\n- Train/test evaluation\n- Confusion matrix\n- Skill extraction\n- Streamlit deployment")
 st.info("The included CSV is a synthetic educational dataset. For production hiring, use a larger, ethically sourced and validated dataset and evaluate fairness.")
