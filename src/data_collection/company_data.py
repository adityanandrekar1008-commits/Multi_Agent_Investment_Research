import yfinance as yf


def get_company_info(ticker):
    stock = yf.Ticker(ticker)

    return stock.info


if __name__ == "__main__":
    company = input("Enter company ticker: ")

    info = get_company_info(company)

    print("Company Name:", info.get("longName"))
    print("Sector:", info.get("sector"))
    print("Industry:", info.get("industry"))
    print("Market Cap:", info.get("marketCap"))
    print("Country:", info.get("country"))