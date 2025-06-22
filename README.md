# PEAD-Strategy

A fully functional backtest of a Post-Earnings Announcement Drift (PEAD) long-short trading strategy. Developed as part of my Bachelor's thesis at the University of Zurich (Major in Banking & Finance, Minor in Computer Science).

## Overview

The strategy systematically trades stocks based on quarterly earnings surprises (SUE scores). It implements:

- Event-based signal generation using SUE thresholds  
- Rule-based long-short execution logic  
- Backtest engine with daily tracking and benchmark comparison (S&P 500)  
- Performance metrics: Sharpe ratio, Jensen’s alpha, Beta, Total and Annualized Return  
- Transaction cost modeling using bid-ask spreads  

## Requirements

- Python 3.8+  
- WRDS access (for data)  
- macOS or Windows (tested on both)  

## Installation & Execution

### macOS / Linux

```bash
./run.sh
```

### Windows

Double-click `run.bat` or execute:

```cmd
run.bat
```

This will:
- Set up a Python virtual environment  
- Install all dependencies  
- Run the full backtest (`src/pead_strategy.py`)  

## Output

After completion, the script:
- Prints a summary of results to the terminal  
- Saves a `.txt` summary in the `output/` directory  
- Shows a plot comparing portfolio vs. benchmark performance  

> The backtest covers the period from 2000 to 2020.  
> The `output/` folder contains the full result summary as a `.txt` file and the equity curve plot comparing the PEAD strategy against the benchmark.

## 📁 Data Disclaimer

The `data/` folder is intentionally left empty due to GitHub file size limits and licensing restrictions.

### Data Sources

The datasets were obtained from four different sources:

- **IBES (via WRDS)**: Quarterly earnings announcement data, including actual EPS, the mean and standard deviation of analyst estimates, announcement dates, and tickers.
- **CRSP (via WRDS)**: Daily adjusted prices, bid and ask prices, and historical S&P 500 constituent membership. Used for returns, transaction cost modeling, and universe construction.
- **Compustat (via WRDS)**: Complementary daily stock prices to fill missing values in CRSP, SIC industry codes for sector analysis.
- **Yahoo Finance**: Daily adjusted close prices for the SPDR S&P 500 ETF Trust (SPY), used as the benchmark.

The full earnings dataset contains 95,748 announcements for 1,337 stocks that have been part of the S&P 500 between 1994 and 2023. Daily prices are adjusted for splits, dividends, and mergers.

> These datasets are sourced from [WRDS](https://wrds-www.wharton.upenn.edu/) and Yahoo Finance. WRDS data is subject to licensing restrictions and must **not** be redistributed.  
> If you are affiliated with a university, you may have access via WRDS.  
> Otherwise, contact the author to request a minimal demo dataset for evaluation purposes.

## License

This project is released under the MIT License (see `LICENSE`).

## Author

**Aladin Bouddat**  
[GitHub](https://github.com/aladinbouddat) • [LinkedIn](https://www.linkedin.com/in/aladinbouddat)
