def remove_duplicates(arr):
    # Use a set to store unique strings
    unique_strings = set()
    
    # Create a new list to store the unique strings in order
    result = []
    
    for string in arr:
        # If the string is not in the set, add it to the set and the result list
        if string not in unique_strings:
            unique_strings.add(string)
            result.append(string)
    
    return result

# Example usage:
stock_symbols = ["SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "AAPL", "AMZN", "GOOG", "META", "TSLA", "NVDA", "MSFT", "JPM", "BAC", "WFC", "C", "V", "MA", "PYPL", "ADBE", "CRM", "NFLX", "DIS", "HD", 
                 "MCD", "NKE", "SBUX", "KO", "PEP", "PG", "JNJ", "UNH", "PFE", "MRK", "ABBV", "CVS", "WMT", "TGT", "COST", "LOW", "HD", "TJX", "M", "AMT", "CCI", "PLD", "SPG", "EQIX", "DLR", 
                 "PSA", "AVB", "EQR", "AIV", "UDR", "VTR", "O", "WY", "BXP", "SLG", "ARE", "HST", "HLT", "MAR", "H", "HGV", "IHG", "CCL", "RCL", "NCLH", "LUV", "UAL", "DAL", "AAL", "JBLU", "ALK", 
                 "SAVE", "EXPE", "BKNG", "TRIP", "LYV", "SIX", "FUN", "PLNT", "SEAS", "CZR", "MGM", "WYNN", "LVS", "TGT", "WMT", "COST", "HD", "LOW", "TJX", "ROST", "BBY", "TSCO", "DG", "DLTR", "KR",
                 "SOXL", "SOXS", "TQQQ", "SQQQ", "FNGU", "FNGD", "SPXL", "SPXS", "UDOW", "SDOW", "TNA", "TZA", "LABU", "LABD", "NUGT", "DUST", "JNUG", "JDST", "GUSH", "DRIP", "ERX", "ERY", "FAS", "FAZ",
                 "UVXY", "TLT", "XLE", "XLF", "XLU", "XLK", "XLI", "XLB", "XLP", "XLV", "XLY", "XBI", "XOP", "XRT", "XHB", "XME", "XSD", "XSW", "XITK", "XNTK", "XWEB",  
                 "BOIL", "USO", "GLDM", "AMD", "INTC", "MU", "QCOM", "TXN", "AVGO", "AMAT", "ADP", "ADSK", "ASML", "BIDU", "BIIB", "BMRN", "CDNS", "CELG", "CERN", "CHKP", "CTAS", "CTSH", "CTXS", "DLTR",
                 "EA", "EBAY", "EXPE", "FAST", "FISV", "GILD", "HAS", "HSIC", "IDXX", "ILMN", "INCY", "INTU", "ISRG", "JBHT", "KLAC", "LRCX", "MAR", "MCHP", "MDLZ", "MNST", "MXIM", "MYL", "NTAP", "NTES",
                 "XOM", "CVX", "GS", "UNP", "RTX", "BA", "MMM", "CAT", "IBM", "HON", "NKE", "MCD", "VZ", "JNJ", "PG", "WMT", "LMT", "COST", "GE", "LLY", "UTX", "ASML", "SMCI", "SCHW", "GDX", "EEM",
                 "TNA", "EWZ", "FXI", "ARM", "LIN", "CSCO", "DHR", "INTU", "UPS", "USP", "BX", "ADP", "TMO", "AMGN", "MDT", "BLK", "PM", "ISRG", "PNC", "AMT", "CCI", "PLD", "SPG", "EQIX", "DLR", "PSA", 
                  "UBER", "CVS", "ABNB", "BABA", "NIO", "SNAP", "TSM", "SQ", "ROKU", "ZM", "DOCU", "CRWD", "NET", "ZS", "OKTA", "MDB", "DDOG", "SNOW", "FSLY", "PINS", "TWLO", "ETSY", "FVRR", "BRK-B",
                  "GOOGL", "CVX", "ACN", "ABT", "TMUS", "COP", "PM", "UPS", "MS", "BMY", "NOW", "SPGI", "BA", "AXP", "LOW", "DE", "TM", "ELV", "SBUX", "NEE", "TJX", "SYK", "BKNG", "GILD", "MDT", 
                  "ISRG", "SCHW", "MMC", "VRTX", "PGR", "CI", "REGN", "MDLZ", "CB", "SLB", "ADI", "ETN", "LRCX", "EOG", "CME", "PANW", "ZTS", "MO", "BDX", "NOC", "BSX", "SNPS", "LUV", "SO", "FI",
                  "WM", "LULU", "FDX", "MNST", "DELL", "MSI", "KHC", "PLTR", "MRNA", "TTWO", "HYG", "IVV", "LQD", "IEF", "SMH", "ARKK", "SOXX", "QUAL", "TZA", "TNA", "XLRE", "XLF", "XLE", "XLI", "XLB",
                  "SDOW", "TSLQ"]


unique_array = remove_duplicates(stock_symbols)
print(unique_array)
