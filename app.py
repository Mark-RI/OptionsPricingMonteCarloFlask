from flask import Flask, render_template, request
from pandas_datareader import data as pdr
from datetime import date, datetime, timedelta
import yfinance as yf
import requests
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    ticker = ""
    strike_price = ""
    option_type = ""
    latest_price = ""
    std_daily = ""
    std_annual = ""
    chart = ""
    mean_option_payoff = ""
    mean_option_payoff_discounted = ""
    matration_date = ""
    days = ""
    t = ""
    if request.method == "POST":
        # Get the ticker symbol from the form
        ticker_input = request.form.get("ticker")
        strike_price = request.form.get("strike")
        option_type = request.form.get("optionType")
        call = True
        if option_type == "call":
            call = True
        else:
            call = False
        date_str = request.form.get("date")
        end = datetime.strptime(date_str, "%Y-%m-%d").date()
        iterations = 2000
        start = datetime.today()
        # Process the text (for example, convert it to uppercase)
        option_type = option_type.upper()
        ticker = ticker_input.upper()
        print("")
        print(ticker_input)
        print("")
        try:
            strike_price = int(strike_price)
        except (ValueError, TypeError):
            # Handle the error — maybe set a default or return a message
            strike_price = None
            print("Invalid strike price input")
        print(strike_price)
        print(type(strike_price))
        print("")
        print(call)
        print("")
        print(end)

        syms = ['DGS30', 'DGS20', 'DGS10', 'DGS5', 'DGS2', 'DGS1MO', 'DGS3MO']
        ir_data = pdr.DataReader(syms, 'fred', datetime(2006, 12,1),datetime.today())
        names = dict(zip(syms, ['30yr', '20yr', '10yr', '5yr', '2yr', '1m', '3m']))
        ir_data = ir_data.rename(columns=names)
        ir_data = ir_data[['3m']]
        last_row = ir_data.iloc[[-1]]
        risk_free_rate = last_row.to_dict(orient='records')[0]
        risk_free_rate = list(risk_free_rate.values())[0]/100
        risk_free_rate_daily = (risk_free_rate + 1)**(1/252) - 1
        print("Risk free rate (Annual):", round(risk_free_rate, 8))
        print("Risk free rate (Daily):", round(risk_free_rate_daily, 8))

        def get_data(ticker, past_days = 180):
            end_h = datetime.today()
            start_h = end_h - timedelta(days=past_days)

            ticker_data = yf.download(ticker, start_h, end_h, auto_adjust=True)
            return ticker_data

        def get_ticker_stats(stock_data, past_trading_days = 90):
            last_trading_day = stock_data.index[-1]
            first_trading_day = stock_data.index[past_trading_days * -1]
            print("90 trading days ago:", first_trading_day.strftime('%A, %Y-%m-%d'))
            print("Last trading day:", last_trading_day.strftime('%A, %Y-%m-%d'))

            stock_data = stock_data.loc[first_trading_day:last_trading_day]
            stock_data = stock_data[['Close']]
            daily_returns = stock_data.pct_change().dropna()
            latest_price = stock_data.iloc[[-1]]
            latest_price = list(latest_price.to_dict(orient='records')[0].values())[0]
        #    print(latest_price)

            daily_returns_dict = daily_returns.to_dict(orient='records')
        #    print(daily_returns_dict)
            i = 0
            sum = 0
            for close in daily_returns_dict:
                sum += list(close.values())[0]
                i += 1
            mean = sum/i
            print("Average daily percentage change in last 90 days:", round(mean, 8))
            c = 0
            for close in daily_returns_dict:
            #    print(mean - list(close.values())[0])
                c += (mean - list(close.values())[0])**2
            variance = (c/i)
            std_daily = (variance)**(1/2)
            std_annual = std_daily * np.sqrt(252)

            print("Daily Variance:", round(variance, 8))
            print("Annual Standard Deviation:", round(std_annual, 8))
            print("Daily Standard Deviation:", round(std_daily, 8))
        #    print("Latest price:", round(latest_price, 2))

            return latest_price, std_daily, std_annual

        def montecarlo(latest_price, std_daily, start, end, iteratons = 1000):
            end_price = []
            plt.figure(figsize=(10, 6))
            for i in range(iteratons):
                days = ((end - start.date()).days)
                t = np.linspace(0, days, int((252/365)*days))
                prices = [latest_price]

                for idx in range(1, len(t)):
                    rand = np.random.normal(0, 1)
                    X = (risk_free_rate_daily - ((std_daily**2)*0.5)) * (1) + std_daily * rand
                #    X = (risk_free_rate - ((std**2)*0.5)) * (1/252) + std_daily * rand
                    prices.append(prices[idx-1] * np.exp(X))

                plt.plot(t, prices)
                end_price.append(prices[-1])
            plt.title("Monte Carlo Simulation of Stock Prices")
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.grid(True)
            plot_path = "static/montecarlo_plot.png"
            plt.savefig(plot_path)
            plt.close()
            return end_price, plot_path

        def price_option(strike_price, end_price_list, start, end, stats, call = True):
            option_payoff = []
            if call == True:
                for i in end_price_list:
                    d = i - strike_price
                    option_payoff.append(max(d, 0))
            else:
                for i in end_price_list:
                    d = strike_price - i
                    option_payoff.append(max(d, 0))
            mean_option_payoff = round(np.mean(option_payoff), 6)
            print("Average option payoff:", mean_option_payoff)
            days = ((end - start.date()).days)
            t = int((252/365)*days)
            mean_option_payoff_discounted = round(mean_option_payoff*np.exp(-risk_free_rate*(t/252)), 6)
            print("Average option payoff discounted:", mean_option_payoff_discounted)
            print("Strike price:", round(strike_price, 2))
            print("Current price:", round(stats[0], 2))
            print("Maturation date:", end)
            print("Current date:", start.date())
            print("Days until maturation:", days)
            print("Trading days until maturation:", t)

            return mean_option_payoff, mean_option_payoff_discounted, end, days, t

        ticker_data = get_data(ticker)
        ticker_stats = get_ticker_stats(ticker_data)
        latest_price = round(ticker_stats[0], 2)
        std_daily = round(ticker_stats[1], 8)
        std_annual = round(ticker_stats[2], 8)
        end_price_ticker = montecarlo(ticker_stats[0], ticker_stats[1], start, end, iterations)
        chart = end_price_ticker[1]
        price_option = price_option(strike_price, end_price_ticker[0], start, end, ticker_stats, call)
        mean_option_payoff = price_option[0]
        mean_option_payoff_discounted = price_option[1]
        matration_date = price_option[2]
        days = price_option[3]
        t = price_option[4]

    return render_template("index.html", ticker=ticker, strike_price=strike_price, latest_price=latest_price, option_type=option_type, std_daily=std_daily, std_annual=std_annual, annual_chart_html=chart, mean_option_payoff=mean_option_payoff, mean_option_payoff_discounted=mean_option_payoff_discounted, matration_date=matration_date, days=days, t=t)