import subprocess
import sys
from tqdm import tqdm  # Import tqdm for a progress bar
import pandas as pd
import os
from datetime import datetime
import concurrent.futures
import shutil

period = "300d"  # Period, this does not effect anything besides txt file naming, make sure to change it in loopFile.py as well
max_workers = 10  # Number of threads to use

path = "loopFile.py"  # Path to the file to be executed

stock_symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'SLV', 'AAPL', 'AMZN', 'GOOG', 'META', 'TSLA', 'NVDA', 'MSFT', 'JPM', 'BAC', 'WFC', 'C', 'V', 'MA', 
                 'PYPL', 'ADBE', 'CRM', 'NFLX', 'DIS', 'HD', 'MCD', 'NKE', 'SBUX', 'KO', 'PEP', 'PG', 'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'CVS', 'WMT', 
                 'TGT', 'COST', 'LOW', 'TJX', 'M', 'AMT', 'CCI', 'PLD', 'SPG', 'EQIX', 'DLR', 'PSA', 'AVB', 'EQR', 'AIV', 'UDR', 'VTR', 'O', 'WY', 'BXP', 
                 'SLG', 'ARE', 'HST', 'HLT', 'MAR', 'H', 'HGV', 'IHG', 'CCL', 'RCL', 'NCLH', 'LUV', 'UAL', 'DAL', 'AAL', 'JBLU', 'ALK', 'SAVE', 'EXPE', 
                 'BKNG', 'TRIP', 'SIX', 'FUN', 'PLNT', 'SEAS', 'CZR', 'MGM', 'WYNN', 'LVS', 'ROST', 'BBY', 'TSCO', 'DG', 'DLTR', 'KR', 'SOXL', 
                 'TQQQ', 'FNGU', 'SPXL', 'UDOW', 'TNA', 'NUGT', 'JNUG',  
                 'GUSH', 'ERX', 'FAS', 'UVXY', 'TLT', 'XLE', 'XLF', 'XLU', 'XLK', 'XLI', 'XLB', 'XLP', 'XLV', 'XLY', 'XBI', 'XOP', 
                 'XRT', 'XHB', 'XME', 'XSD', 'XSW', 'XITK', 'XNTK', 'XWEB', 'BOIL', 'USO', 'GLDM', 'AMD', 'INTC', 'MU', 'QCOM', 'TXN', 'AVGO', 'AMAT', 'ADP', 
                 'ADSK', 'ASML', 'BIDU', 'BIIB', 'BMRN', 'CDNS', 'CELG', 'CERN', 'CHKP', 'CTAS', 'CTSH', 'CTXS', 'EA', 'EBAY', 'FAST', 'GILD', 'HAS', 
                 'HSIC', 'IDXX', 'ILMN', 'INCY', 'INTU', 'ISRG', 'JBHT', 'KLAC', 'LRCX', 'MCHP', 'MDLZ', 'MNST', 'MXIM', 'MYL', 'NTAP', 'NTES', 'XOM', 'CVX', 'GS', 
                 'UNP', 'RTX', 'BA', 'MMM', 'CAT', 'IBM', 'HON', 'VZ', 'LMT', 'GE', 'LLY', 'SMCI', 'SCHW', 'GDX', 'EEM', 'EWZ', 'FXI', 'ARM', 'LIN', 'CSCO', 
                 'DHR', 'UPS', 'USP', 'BX', 'TMO', 'AMGN', 'MDT', 'BLK', 'PM', 'PNC', 'UBER', 'ABNB', 'BABA', 'NIO', 'SNAP', 'TSM', 'SQ', 'ROKU', 'ZM', 'DOCU', 'CRWD', 
                 'NET', 'ZS', 'OKTA', 'MDB', 'DDOG', 'SNOW', 'FSLY', 'PINS', 'TWLO', 'ETSY', 'FVRR', 'BRK-B', 'GOOGL', 'ACN', 'ABT', 'TMUS', 'COP', 'MS', 'BMY', 'NOW', 
                 'SPGI', 'AXP', 'DE', 'TM', 'ELV', 'NEE', 'SYK', 'MMC', 'VRTX', 'PGR', 'CI', 'REGN', 'CB', 'SLB', 'ADI', 'ETN', 'EOG', 'CME', 'PANW', 'ZTS', 'MO', 'BDX', 
                 'NOC', 'BSX', 'SNPS', 'SO', 'FI', 'WM', 'LULU', 'FDX', 'DELL', 'MSI', 'KHC', 'PLTR', 'MRNA', 'TTWO', 'HYG', 'IVV', 'LQD', 'IEF', 'SMH', 'ARKK', 'SOXX', 
                 'QUAL', 'XLRE', 'MSTR']
allProfitArray = []
allBuyPriceArray = []
allBuyDateArray = []

# Define a function to execute the loopFile.py file for a given stock symbol
def process_stock_symbol(stock_symbol):
    result = subprocess.run([sys.executable, path, stock_symbol], capture_output=True, text=True)  # Execute the file with the stock symbol as an argument
    print("Finished processing " + stock_symbol)  # Print a message when the file is finished processing

    # Check the exit code to see if file 2 ran successfully
    if result.returncode == 0:
        output_lines = result.stdout.splitlines()

        # Iterate through each line of the subprocess output
        if "Buy Price:" in output_lines[0]:
            for line in output_lines:
                if "Buy Price:" in line:
                    buy_price = line.split("Buy Price: ")[1].split(" Buy Date:")[0]
                    buy_date = line.split("Buy Date: ")[1]
                    allBuyPriceArray.append(buy_price)
                    allBuyDateArray.append(buy_date)
                    print("Buy Price:", buy_price, "Buy Date:", buy_date)


                if "Total Profit:" in line:
                    total_profit = float(line.split("Total Profit: ")[1])
                    allProfitArray.append(total_profit)
                    print("Total Profit:", total_profit)


        elif "No open position" in output_lines[0]:
            print("No open position")
            buy_price = None
            buy_date = None
            allBuyPriceArray.append(buy_price)
            allBuyDateArray.append(buy_date)

            total_profit = float(output_lines[2].split(":")[1])
            allProfitArray.append(total_profit)
            print("Total Profit:", total_profit)
        else:
            print("Unrecognized output format")
    else:
        print("File 2 encountered an error or didn't run successfully.")

# Use a thread pool to execute the process_stock_symbol function for each stock symbol
with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
    futures = [executor.submit(process_stock_symbol, stock_symbol) for stock_symbol in stock_symbols]
    for future in concurrent.futures.as_completed(futures):
        pass

# Create a dictionary with the data
data = {
    'Stock Symbol': stock_symbols,
    'Profit': allProfitArray,
    'Buy Price': allBuyPriceArray,
    'Buy Date': allBuyDateArray
}

# Create a Pandas DataFrame from the dictionary
df = pd.DataFrame(data)

# Convert the 'Buy Date' column to a datetime type, so we can sort it
df['Buy Date'] = pd.to_datetime(df['Buy Date'], errors='coerce')

# Sort the DataFrame by 'Buy Date' in descending order
df = df.sort_values(by='Buy Date', ascending=False)

#df = df.sort_values(by='Profit', ascending=False)

# Reset the index to start from 0
df = df.reset_index(drop=True)

# Save the sorted DataFrame to a properly formatted text file
with open("output.txt", "w") as f:
    f.write(df.to_string(index=False))


# ...

# Save the sorted DataFrame to a properly formatted text file
with open("output.txt", "w") as f:
    f.write(df.to_string(index=False))

# Create directories if they don't exist
if not os.path.exists("historicalLoopData"):
    os.mkdir("historicalLoopData")
if not os.path.exists(f"historicalLoopData/{period}"):
    os.mkdir(f"historicalLoopData/{period}")

# Create a filename using the current date and time
now = datetime.now()
filename = now.strftime("%Y-%m-%d.txt")

# Copy the output.txt file to the subdirectory with the new filename
shutil.copy("output.txt", f"historicalLoopData/{period}/{filename}")
