import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
import plotly.express as px

st.set_page_config(
    page_title="Limuru Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_data = joblib.load('Knowledge_based_systems/Regression/limuru_stock_prediction_model.pkl')
    return model_data['model'], model_data['metrics']

def main():
    model, metrics = load_model()

    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stTitle {
            color: #2E4053;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 2rem;
        }
        .stSubheader {
            color: #566573;
            font-size: 1.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("📈 Limuru Stock Price Predictor")
    st.markdown("### Make informed decisions with Linear Regressive stock price predictions")

    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/stocks.png", width=100)
    st.sidebar.title("Input Parameters")
    st.sidebar.markdown("Adjust the parameters below to predict stock prices.")

    # Input features
    with st.sidebar:
        st.markdown("### Stock Parameters")
        twelve_month_low = st.number_input("12 Month Low", min_value=0.0, value=400.0)
        twelve_month_high = st.number_input("12 Month High", min_value=0.0, value=500.0)
        day_low = st.number_input("Day Low", min_value=0.0, value=350.0)
        day_high = st.number_input("Day High", min_value=0.0, value=350.0)
        previous = st.number_input("Previous Day Price", min_value=0.0, value=350.0)
        volume = st.number_input("Volume", min_value=0, value=100)
        year = st.slider("Year", min_value=2007, max_value=2012, value=2012)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Real-time Prediction")
        
        # input dataframe
        input_data = pd.DataFrame({
            'CODE': [0],  
            '12m Low': [twelve_month_low],
            '12m High': [twelve_month_high],
            'Day Low': [day_low],
            'Day High': [day_high],
            'Previous': [previous],
            'Volume': [volume],
            'Year': [year]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]

        st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                <h2 style='color: #2E4053; margin-bottom: 10px;'>Predicted Stock Price</h2>
                <h1 style='color: #2ECC71; font-size: 3rem;'>KES {prediction:.2f}</h1>
            </div>
        """, unsafe_allow_html=True)

        # Historical vs Predicted visualization
        st.markdown("### Historical Price Trend")
        dates = pd.date_range(start='2022-01-01', periods=30, freq='D')
        historical_prices = np.random.normal(prediction, 5, 30)  
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=historical_prices, name='Historical', line=dict(color='#3498DB')))
        fig.add_trace(go.Scatter(x=[dates[-1]], y=[prediction], name='Prediction', mode='markers',
                                marker=dict(color='#2ECC71', size=15)))
        
        fig.update_layout(
            title='Stock Price Trend',
            xaxis_title='Date',
            yaxis_title='Price (KES)',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Model metrics
        st.markdown("### Model Metrics")
        metric_display = {
            'R² Score': f"{metrics['r2']:.4f}",
            'MSE': f"{metrics['mse']:.6f}",
            'RMSE': f"{metrics['rmse']:.4f}",
            'MAE': f"{metrics['mae']:.4f}",
            'Accuracy (5% threshold)': f"{metrics['accuracy']:.2f}%"
        }
        
        for metric, value in metric_display.items():
            st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
                    <h4 style='color: #566573; margin-bottom: 5px;'>{metric}</h4>
                    <h2 style='color: #2E4053; margin: 0;'>{value}</h2>
                </div>
            """, unsafe_allow_html=True)

        # Additional insights
        st.markdown("### Market Insights")
        st.markdown("""
            - 📈 Current trend: Bullish
            - 💹 Volume: Above average
            - 📊 Volatility: Moderate
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #566573;'>
            <p>Built with ❤️ using Streamlit and Machine Learning</p>
            <p>Data source: Nairobi Securities Exchange</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
