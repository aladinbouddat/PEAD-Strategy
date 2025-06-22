# PEAD-Strategy

A quantitative trading strategy exploiting the Post-Earnings Announcement Drift (PEAD) anomaly. Developed as part of my Bachelor's thesis at the University of Zurich (BSc Banking & Finance, Minor in Computer Science).

## Overview

This strategy detects statistically significant earnings surprises (SUE scores) and runs a rule-based long-short portfolio. Features include:

- Event-driven signal generation  
- Trade execution with realistic transaction cost modeling  
- Daily portfolio tracking and benchmark (S&P 500) comparison  
- Performance metrics:
  - Total and annualized return  
  - Sharpe ratio, Jensen’s alpha, Beta, Excess return

---

## Quick Start

### Prerequisites

- Python 3.7 or higher
- CSV input files placed in the `data/` directory (not included in this repo)

---

### macOS / Linux

```bash
git clone https://github.com/uzhprogrammer/PEAD-strategy.git
cd PEAD-strategy
chmod +x run.sh
./run.sh
```

---

### Windows

1. Clone or download this repository  
2. Place the required `.csv` files into the `data/` folder  
3. Double-click `run.bat` to:
   - Create a virtual environment (if missing)  
   - Install dependencies (on first run)  
   - Launch the strategy

If it opens in a text editor: right-click → "Open with" → Command Prompt

---

## Output

- `output/backtest_summary.txt` – Plain-text summary of results  
- `output/plot_<year>.png` – Equity curve of portfolio vs. benchmark

Includes:
- Total and annualized return  
- Volatility  
- Sharpe ratio, Alpha, Beta  
- Transaction cost breakdown

---

## Why This Project

This project demonstrates my ability to:

- Translate financial research into efficient Python pipelines  
- Build realistic backtest engines with daily tracking  
- Work with real-market data and evaluate risk-adjusted returns

---

## Author

**Aladin Bouddat**  
[GitHub](https://github.com/uzhprogrammer)  
[LinkedIn](https://www.linkedin.com/in/aladinbouddat)
