import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="🏠 AI House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

 
st.markdown("""
<style>

.main{
    background-color:#f4f7fb;
}

h1{
    color:#0F172A;
    text-align:center;
    font-weight:800;
}

h3{
    color:#1E40AF;
}

.stButton>button{
    width:100%;
    background:linear-gradient(90deg,#2563eb,#0ea5e9);
    color:white;
    border-radius:12px;
    border:none;
    padding:12px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:linear-gradient(90deg,#1d4ed8,#0284c7);
    transform:scale(1.02);
}

.metric-card{
background:white;
padding:18px;
border-radius:15px;
box-shadow:0px 0px 12px rgba(0,0,0,0.12);
text-align:center;
margin-bottom:20px;
}

.sidebar .sidebar-content{
background:#ffffff;
}

</style>
""", unsafe_allow_html=True)

 

st.image(
"https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=1600",
use_container_width=True
)

st.title("🏠 AI House Price Prediction Dashboard")

st.write(
"""
Predict residential house prices using **Multiple Linear Regression**.

Model Features:

- Bedrooms
- Bathrooms
- Square Feet
- Age of House
"""
)

# -----------------------
# DATASET
# -----------------------

data = {

"Bedrooms":[2,3,4,3,5,4,2,3,4,5,
3,2,4,3,5,4,2,3,4,5,
2,3,4,3,5,4,2,3,4,5],

"Bathrooms":[1,2,3,2,4,3,1,2,3,4,
2,1,3,2,4,3,1,2,3,4,
1,2,3,2,4,3,1,2,3,4],

"Square_Feet":[900,1200,1800,1400,2800,
2200,950,1350,1850,3000,
1450,1000,2100,1500,3200,
2400,980,1250,1950,2900,
1050,1600,2000,1700,3100,
2300,1100,1550,2150,3300],

"Age_of_House":[18,12,8,10,3,
5,22,15,7,2,
13,20,6,11,1,
4,24,17,5,2,
19,9,7,10,1,
3,21,14,6,2],

"Price":[4200000,6100000,9300000,7100000,15800000,
12200000,4300000,6400000,9800000,17000000,
7000000,4500000,11800000,7600000,17600000,
12800000,4400000,6200000,10100000,16500000,
4700000,7800000,11000000,8200000,18000000,
13500000,4800000,7900000,12000000,18800000]

}

df = pd.DataFrame(data)

# -----------------------
# MODEL
# -----------------------

X = df[[
"Square_Feet",
"Bedrooms",
"Bathrooms",
"Age_of_House"
]]

y = df["Price"]

model = LinearRegression()

model.fit(X,y)

pred_train = model.predict(X)

r2 = r2_score(y,pred_train)

mae = mean_absolute_error(y,pred_train)

rmse = np.sqrt(mean_squared_error(y,pred_train))

 

st.sidebar.header("Enter House Details")

sqft = st.sidebar.slider(
"Square Feet",
800,
3500,
1500,
50
)

bedrooms = st.sidebar.slider(
"Bedrooms",
1,
6,
3
)

bathrooms = st.sidebar.slider(
"Bathrooms",
1,
5,
2
)

age = st.sidebar.slider(
"Age of House",
0,
30,
10
)

predict = st.sidebar.button("🏠 Predict Price")

 
# PREDICTION SECTION
 

st.markdown("---")
st.subheader("🔮 House Price Prediction")

input_df = pd.DataFrame({
    "Square_Feet":[sqft],
    "Bedrooms":[bedrooms],
    "Bathrooms":[bathrooms],
    "Age_of_House":[age]
})

predicted_price = model.predict(input_df)[0]

 
# BUTTON ACTION
 

if predict:

    st.balloons()

    st.success("Prediction Completed Successfully!")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
        f"""
        <div class="metric-card">
        <h3>💰 Predicted Price</h3>
        <h2 style="color:green;">
        ₹ {predicted_price:,.0f}
        </h2>
        </div>
        """,
        unsafe_allow_html=True
        )

    with col2:

        st.markdown(
        f"""
        <div class="metric-card">
        <h3>📐 Square Feet</h3>
        <h2>{sqft}</h2>
        </div>
        """,
        unsafe_allow_html=True
        )

    with col3:

        st.markdown(
        f"""
        <div class="metric-card">
        <h3>🛏 Bedrooms</h3>
        <h2>{bedrooms}</h2>
        </div>
        """,
        unsafe_allow_html=True
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        st.markdown(
        f"""
        <div class="metric-card">
        <h3>🛁 Bathrooms</h3>
        <h2>{bathrooms}</h2>
        </div>
        """,
        unsafe_allow_html=True
        )

    with col5:

        st.markdown(
        f"""
        <div class="metric-card">
        <h3>🏡 Age</h3>
        <h2>{age} Years</h2>
        </div>
        """,
        unsafe_allow_html=True
        )

    with col6:

        st.markdown(
        f"""
        <div class="metric-card">
        <h3>📈 Model Accuracy</h3>
        <h2>{r2*100:.2f}%</h2>
        </div>
        """,
        unsafe_allow_html=True
        )

 
# MODEL PERFORMANCE
 

st.markdown("---")

st.subheader("📊 Model Performance")

m1,m2,m3 = st.columns(3)

m1.metric("R² Score",f"{r2:.4f}")

m2.metric("MAE",f"₹ {mae:,.0f}")

m3.metric("RMSE",f"₹ {rmse:,.0f}")

 
# MODEL COEFFICIENTS
 

coef = pd.DataFrame({

    "Feature":X.columns,

    "Coefficient":model.coef_

})

st.subheader("📌 Feature Importance (Linear Regression Coefficients)")

st.dataframe(
    coef,
    use_container_width=True
)

# ============================================================
# VISUALIZATIONS
# ============================================================

st.markdown("---")
st.subheader("📈 Actual vs Predicted House Prices")

# Create dataframe for visualization
results = pd.DataFrame({
    "Actual Price": y,
    "Predicted Price": pred_train,
    "Square Feet": df["Square_Feet"],
    "Bedrooms": df["Bedrooms"],
    "Bathrooms": df["Bathrooms"],
    "Age": df["Age_of_House"]
})

# Scatter Plot
fig = px.scatter(
    results,
    x="Actual Price",
    y="Predicted Price",
    hover_data=["Square Feet","Bedrooms","Bathrooms","Age"],
    title="Actual Price vs Predicted Price",
    color="Bedrooms",
    size="Square Feet",
    template="plotly_white"
)

# Perfect Prediction Line
min_price = min(results["Actual Price"].min(),
                results["Predicted Price"].min())

max_price = max(results["Actual Price"].max(),
                results["Predicted Price"].max())

fig.add_trace(

    go.Scatter(

        x=[min_price,max_price],

        y=[min_price,max_price],

        mode="lines",

        name="Perfect Prediction",

        line=dict(color="red",dash="dash")

    )

)

# Highlight User Prediction
fig.add_trace(

    go.Scatter(

        x=[predicted_price],

        y=[predicted_price],

        mode="markers+text",

        text=["Your Prediction"],

        textposition="top center",

        marker=dict(

            color="gold",

            size=20,

            symbol="star"

        ),

        name="Your Prediction"

    )

)

fig.update_layout(

    height=650,

    title_x=0.5,

    font=dict(size=15)

)

st.plotly_chart(fig,use_container_width=True)

 
# FEATURE RELATIONSHIP
 

st.markdown("---")

st.subheader("🏠 Square Feet vs House Price")

fig2 = px.scatter(

    df,

    x="Square_Feet",

    y="Price",

    color="Bedrooms",

    size="Bathrooms",

    trendline="ols",

    template="plotly_white",

    title="Relationship Between Square Feet and Price"

)

fig2.update_layout(height=600)

st.plotly_chart(fig2,use_container_width=True)

 
# PRICE DISTRIBUTION
 

st.markdown("---")

st.subheader("💰 House Price Distribution")

fig3 = px.histogram(

    df,

    x="Price",

    nbins=10,

    color_discrete_sequence=["royalblue"],

    template="plotly_white"

)

fig3.update_layout(height=500)

st.plotly_chart(fig3,use_container_width=True)

 
# FEATURE CORRELATION
 

st.markdown("---")

st.subheader("🔥 Correlation Heatmap")

corr = df.corr(numeric_only=True)

heat = go.Figure(

    data=go.Heatmap(

        z=corr.values,

        x=corr.columns,

        y=corr.columns,

        colorscale="Blues",

        text=np.round(corr.values,2),

        texttemplate="%{text}",

        hoverongaps=False

    )

)

heat.update_layout(height=600)

st.plotly_chart(heat,use_container_width=True)

 
# DATASET PREVIEW
 

st.markdown("---")
st.subheader("📋 Sample Housing Dataset")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

 
# DATASET STATISTICS
 

st.markdown("---")
st.subheader("📊 Dataset Statistics")

st.dataframe(
    df.describe(),
    use_container_width=True
)

 
# PREDICTION SUMMARY
 

st.markdown("---")
st.subheader("📝 Prediction Summary")

summary = pd.DataFrame({

    "Feature":[
        "Square Feet",
        "Bedrooms",
        "Bathrooms",
        "Age of House",
        "Predicted Price"
    ],

    "Value":[
        sqft,
        bedrooms,
        bathrooms,
        age,
        f"₹ {predicted_price:,.0f}"
    ]

})

st.table(summary)

# ============================================================
# DOWNLOAD PREDICTION
# ============================================================

download_df = pd.DataFrame({

    "Square_Feet":[sqft],
    "Bedrooms":[bedrooms],
    "Bathrooms":[bathrooms],
    "Age_of_House":[age],
    "Predicted_Price":[predicted_price]

})

csv = download_df.to_csv(index=False).encode("utf-8")

st.download_button(

    label="📥 Download Prediction CSV",

    data=csv,

    file_name="House_Price_Prediction.csv",

    mime="text/csv"

)

 
# ABOUT MODEL
 

st.markdown("---")
st.subheader("🤖 About This Model")

st.info("""
This dashboard uses **Multiple Linear Regression** from **scikit-learn**.

Input Features:
- Square Feet
- Bedrooms
- Bathrooms
- Age of House

Target:
- House Price

The model learns the relationship between these features and predicts the price of a house based on your inputs.
""")

 
# FOOTER
 

st.markdown("---")

st.markdown(
"""
<div style="text-align:center;
padding:18px;
background:#0F172A;
color:white;
border-radius:12px;">

<h3>🏠 AI House Price Prediction Dashboard</h3>

<p>Built with ❤️ using Python, Streamlit, Scikit-Learn & Plotly</p>

<p>Machine Learning Model: Multiple Linear Regression</p>

</div>
""",
unsafe_allow_html=True
)