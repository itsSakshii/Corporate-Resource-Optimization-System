
import streamlit as st
import requests
import pandas as pd
import altair as alt


st.set_page_config(page_title="Corporate Resource Optimization", layout="wide")

st.title("🏢 Corporate Resource Optimization Dashboard")
st.markdown("Predict employee performance based on workload and other parameters.")


tab_personal, tab_work, tab_performance = st.tabs(["👤 Personal", "💼 Work", "📊 Performance"])

with tab_personal:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 
            min_value=22, max_value=60, value=41,
            help="Range: 22-60 years (mean: 41)"
        )
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        education = st.selectbox("Education Level", ["Bachelor", "High School", "Master", "PhD"], index=0)
    with col2:
        resigned = st.selectbox("Resigned", ["No", "Yes"])  # will be converted to boolean when sent
        promotions = st.number_input("Promotions", 
            min_value=0, max_value=10, value=0,
            help="Number of career promotions"
        )

with tab_work:
    col1, col2 = st.columns(2)
    with col1:
        department = st.selectbox("Department", [
            "Customer Support",
            "Engineering",
            "Finance",
            "HR",
            "IT",
            "Legal",
            "Marketing",
            "Operations",
            "Sales",
        ])
        years_at_company = st.number_input("Years at Company", 
            min_value=0, max_value=10, value=4,
            help="Range: 0-10 years (mean: 4.5)"
        )
        hours = st.number_input("Work Hours Per Week", 
            min_value=30, max_value=60, value=45,
            help="Range: 30-60 hours/week (mean: 45)"
        )
        overtime = st.number_input("Overtime Hours", 
            min_value=0, max_value=29, value=15,
            help="Range: 0-29 hours (mean: 14.5)"
        )
        sick_days = st.number_input("Sick Days Taken", 
            min_value=0, max_value=14, value=7,
            help="Range: 0-14 days (mean: 7)"
        )
    with col2:
        budget = st.number_input("Monthly Salary", 
            min_value=3850, max_value=9000, value=6400,
            help="Range: 3,850-9,000 (mean: 6,400). Strong predictor of performance."
        )
        team_size = st.number_input("Team Size", 
            min_value=1, max_value=19, value=10,
            help="Range: 1-19 members (mean: 10)"
        )
        projects = st.number_input("Projects Handled",
            min_value=0, max_value=49, value=24,
            help="Range: 0-49 projects (mean: 24)"
        )
        remote_freq = st.number_input("Remote Work Frequency",
            min_value=0, max_value=100, value=50,
            help="Range: 0-100% (mean: 50%)"
        )
        training = st.number_input("Training Hours",
            min_value=0, max_value=99, value=50,
            help="Range: 0-99 hours (mean: 50)"
        )

with tab_performance:
    col1, col2 = st.columns(2)
    with col1:
        satisfaction = st.number_input("Employee Satisfaction Score",
            min_value=1, max_value=100, value=75,
            help="Employee satisfaction rating (1-100)"
        )


if st.button("🚀 Predict Performance"):
    
    input_data = {
        # Personal
        "Age": float(age),
        "Gender": gender,
        "Education_Level": education,
        "Resigned": True if resigned == "Yes" else False,
        "Promotions": int(promotions),
        
        # Work
        "Department": department,
        "Years_At_Company": float(years_at_company),
        "Work_Hours_Per_Week": float(hours),
        "Monthly_Salary": float(budget),
        "Team_Size": int(team_size),
        "Projects_Handled": int(projects),
        "Overtime_Hours": float(overtime),
        "Sick_Days": float(sick_days),
        "Remote_Work_Frequency": float(remote_freq),
        "Training_Hours": float(training),
        
        # Performance
        "Employee_Satisfaction_Score": float(satisfaction)
    }

    try:
        response = requests.post("http://127.0.0.1:5002/predict", json=input_data, timeout=6)

        
        if not response.ok:
            
            try:
                err = response.json()
                st.error(f"⚠️ API Error ({response.status_code}): {err.get('message') or err}")
            except Exception:
                st.error(f"⚠️ API Error ({response.status_code}): {response.text}")
        else:
            
            try:
                result = response.json()
            except ValueError as e:
                st.error(f"❌ Connection error: invalid JSON response: {e}\nResponse text: {response.text}")
            else:
                if result.get("status") == "success" and "prediction" in result:
                    st.success(f"✅ Predicted Performance Score: {result['prediction']}")
                else:
                    st.error(f"⚠️ API Error: {result.get('message') or result}")

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Connection error: {e}")


st.markdown("---")
st.subheader("📈 Example Past Data (Simulated)")
example_data = pd.DataFrame({
    "Department": [1, 2, 3, 2, 1],
    "Work_Hours_Per_Week": [40, 38, 42, 36, 45],
    "Monthly_Salary": [60000, 75000, 55000, 68000, 72000],
    "Performance_Score": [3, 4, 5, 3, 4]
})
st.dataframe(example_data)

try:
    chart = (
        alt.Chart(example_data)
        .mark_bar()
        .encode(x="Work_Hours_Per_Week", y="Performance_Score", color="Department:N")
        .properties(width=600, height=300)
    )
    st.altair_chart(chart)
except Exception as e:
    st.warning(f"Visualization Error: {e}")
