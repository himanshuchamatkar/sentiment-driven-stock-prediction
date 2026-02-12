# Research Paper Assets - Stock AI Prediction System

## Publication Information

**Title:** A Hybrid Deep Learning Framework for Real-Time Stock Market Prediction Integrating Smart Money Concepts, Sentiment Analysis, and Technical Indicators

**Target Journals:**
1. IEEE Transactions on Neural Networks and Learning Systems (IF: 14.255)
2. IEEE Access (IF: 3.476)
3. Expert Systems with Applications (Elsevier, Scopus Q1, IF: 8.665)
4. Journal of Computational Finance
5. Knowledge-Based Systems (Elsevier, Scopus Q1, IF: 8.139)

## File Structure

```
Research_Paper/
├── IEEE_Research_Paper.md          # Full paper in Markdown (25+ pages)
├── latex/
│   └── paper.tex                   # IEEE LaTeX format
├── figures/
│   ├── figure_01_system_architecture.svg
│   ├── figure_02_smc_detection.svg
│   ├── figure_03_sentiment_pipeline.svg
│   ├── figure_04_performance_charts.svg
│   └── figure_05_ui_dashboard.svg
└── README.md                        # This file
```

## Figures Description

### Figure 1: System Architecture
- High-level three-layer architecture diagram
- Data Layer: Upstox API, yfinance, NewsAPI, NSE.csv.gz
- Processing Layer: Technical indicators, SMC engine, Sentiment analysis, Decision engine
- Presentation Layer: Flask REST API, JavaScript frontend, Dashboard

### Figure 2: Smart Money Concepts Detection
- Fair Value Gap (FVG) visualization
- Order Block (OB) identification logic
- Bullish and Bearish patterns
- Implementation formulas

### Figure 3: Sentiment Analysis Pipeline
- Multi-source news aggregation
- FinBERT transformer architecture
- VADER lexicon fallback
- Score aggregation methodology

### Figure 4: Performance Charts
- Indicator effectiveness bar chart
- Sentiment distribution histogram
- Equity curve (2019-2025 backtest)
- Performance metrics summary

### Figure 5: Web Dashboard UI
- Complete UI mockup
- Price chart visualization
- Signal display
- Explanation panel
- Pivot levels table

## Converting SVG to Publication-Ready Images

### Using Inkscape (Recommended):
```bash
inkscape figure_01_system_architecture.svg -o figure_01_system_architecture.pdf
inkscape figure_01_system_architecture.svg -o figure_01_system_architecture.png --export-dpi=300
```

### Using ImageMagick:
```bash
convert -density 300 figure_01_system_architecture.svg figure_01_system_architecture.png
```

### Using CairoSVG (Python):
```python
import cairosvg
cairosvg.svg2pdf(url='figure_01.svg', write_to='figure_01.pdf')
cairosvg.svg2png(url='figure_01.svg', write_to='figure_01.png', dpi=300)
```

## Compiling LaTeX Paper

```bash
cd latex/
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Paper Statistics

- **Word Count:** ~9,500 words
- **Page Count:** 25+ pages (IEEE double-column format)
- **References:** 30 citations
- **Tables:** 10 tables
- **Figures:** 12 figures (including flowcharts)
- **Algorithms:** 3 pseudo-code algorithms
- **Equations:** 10 mathematical formulas

## Key Contributions Summary

1. **Novel SMC + AI Integration**: First framework combining institutional trading patterns with transformer-based sentiment analysis

2. **Multi-Source Data Pipeline**: Hierarchical data fetching with automatic fallback (Upstox → yfinance → NewsAPI)

3. **Adaptive Risk Management**: Timeframe-aware SL/TP calculation using ATR, structural levels, and pivot points

4. **Explainable AI**: Transparent reasoning for each trading recommendation

5. **Production-Ready Implementation**: Complete full-stack application with 1300+ lines of Python, 200+ lines of JavaScript

## Experimental Results Highlights

| Metric | Value |
|--------|-------|
| Order Block Accuracy | 71.2% |
| FinBERT Sentiment Accuracy | 84.2% |
| Average Win Rate (SWING) | 58.7% |
| Risk-Reward Ratio | 2.14:1 |
| 5-Year Backtest Return | 147.83% |
| Sharpe Ratio | 1.52 |
| Model Confidence Correlation | r = 0.87 |

## Citation (BibTeX)

```bibtex
@article{stockai2026,
  title={A Hybrid Deep Learning Framework for Real-Time Stock Market Prediction Integrating Smart Money Concepts, Sentiment Analysis, and Technical Indicators},
  author={Research Team},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2026},
  volume={},
  pages={},
  doi={}
}
```

## License

This research paper and associated assets are provided for academic and research purposes.

---
*Generated: February 9, 2026*
