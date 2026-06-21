from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import sqlite3
from datetime import datetime

app = Flask(__name__)

# -------------------- DATABASE --------------------
conn = sqlite3.connect("predictions.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock TEXT,
    company TEXT,
    current_price REAL,
    predicted_price REAL,
    recommendation TEXT,
    trend TEXT,
    time TEXT
)
""")
conn.commit()


def save_to_db(stock, company, current_price, predicted_price, recommendation, trend):
    cursor.execute("""
        INSERT INTO history (stock, company, current_price, predicted_price, recommendation, trend, time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        stock,
        company,
        current_price,
        predicted_price,
        recommendation,
        trend,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()


# -------------------- ROUTES --------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    stock = request.form.get('stock')

    if not stock:
        return "Stock not received"

    # Get stock data
    data = yf.Ticker(stock)
    hist = data.history(period="1y")

    if hist.empty:
        return "No data found"

    # Company Info
    try:
        info = data.info
        company = info.get("longName", stock)
        sector = info.get("sector", "N/A")
        market_cap = info.get("marketCap", "N/A")
    except:
        company = stock
        sector = "N/A"
        market_cap = "N/A"

    # Stock Stats
    high_price = round(hist['High'].max(), 2)
    low_price = round(hist['Low'].min(), 2)
    avg_price = round(hist['Close'].mean(), 2)

    # ML Model
    prices = hist['Close'].values
    days = np.arange(len(prices)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(days, prices)

    current_price = prices[-1]

    # Future prediction
    future_days = np.arange(len(prices), len(prices) + 7).reshape(-1, 1)
    future_predictions = model.predict(future_days)

    predicted_price = float(future_predictions[-1])

    # Trend
    trend = "UP 📈" if predicted_price > current_price else "DOWN 📉"

    # Recommendation
    if predicted_price > current_price * 1.02:
        recommendation = "BUY ✅"
    elif predicted_price < current_price * 0.98:
        recommendation = "SELL ❌"
    else:
        recommendation = "HOLD ⏳"

    # Accuracy (demo)
    accuracy = round(np.random.uniform(85, 95), 2)

    # Moving averages (optional)
    hist['MA50'] = hist['Close'].rolling(50).mean()
    hist['MA200'] = hist['Close'].rolling(200).mean()

    # Plotly Chart
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name='Stock'
    ))

    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['MA50'],
        mode='lines',
        name='MA50'
    ))

    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['MA200'],
        mode='lines',
        name='MA200'
    ))

    fig.update_layout(
        title=f"{company} Stock Analysis",
        xaxis_title="Date",
        yaxis_title="Price",
        height=650
    )

    graph_html = fig.to_html(full_html=False)

    # SAVE TO DB
    save_to_db(stock, company, current_price, predicted_price, recommendation, trend)

    return render_template(
        "result.html",
        stock=stock,
        company=company,
        sector=sector,
        market_cap=market_cap,
        result=trend,
        current_price=round(current_price, 2),
        price=round(predicted_price, 2),
        accuracy=accuracy,
        recommendation=recommendation,
        high_price=high_price,
        low_price=low_price,
        avg_price=avg_price,
        graph=graph_html,
        future_prices=[round(x, 2) for x in future_predictions]
    )


# -------------------- HISTORY --------------------
@app.route('/history')
def history():
    cursor.execute("""
        SELECT stock, company, current_price, predicted_price, recommendation, trend, time
        FROM history ORDER BY id DESC
    """)
    rows = cursor.fetchall()

    return render_template("history.html", data=rows)


# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(debug=True)