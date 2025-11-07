# Anomaly Validator - LLM Multi-Agent Framework

**Based on:** "Enhancing Anomaly Detection in Financial Markets with an LLM-Based Multi-Agent Framework" (Park, 2024)

## Overview

This module validates anomalies detected by the Anomaly_Detector (Paper 1) using a multi-agent LLM system. It:
1. Analyzes supporting documents for relevance
2. Performs web scraping for external context
3. Uses AI reasoning to validate or unflag anomalies
4. Generates natural language explanations

## Architecture

```
Anomaly → Document Agent → Research Agent → Reasoning Agent → Report Agent → Validation Result
```

### Agents

- **Document Agent**: Retrieves and assesses supporting documents using Gemini
- **Research Agent**: Web scraping (news, company sites, regulatory filings)
- **Reasoning Agent**: Validates anomalies with AI reasoning
- **Report Agent**: Generates NLP explanations and summaries

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API keys:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

3. (Optional) Install Tesseract for OCR:
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

## Usage

### Validate Single Anomaly
```python
from src.pipeline.validator_pipeline import AnomalyValidatorPipeline

pipeline = AnomalyValidatorPipeline()
result = pipeline.validate_anomaly(anomaly_record)
print(result.explanation)
```

### Full Pipeline (Detection + Validation)
```bash
python scripts/run_full_pipeline.py --config config/validator_config.yaml
```

## Configuration

Edit `config/validator_config.yaml` to customize:
- LLM model and parameters
- Document processing settings
- Web scraping sources
- Validation thresholds

## Directory Structure

```
Anomaly_Validator/
├── src/
│   ├── agents/              # Multi-agent system
│   ├── document_processing/ # Document parsers
│   ├── web_scraping/        # Web scrapers
│   ├── llm/                 # Gemini integration
│   └── pipeline/            # Orchestration
├── scripts/                 # Executable scripts
├── data/                    # Data storage
├── config/                  # Configuration files
└── tests/                   # Unit tests
```

## Examples

See `scripts/test_validator.py` for usage examples.

## Integration with Anomaly_Detector

The validator automatically processes anomalies from the detection queue:
```
Anomaly_Detector → anomaly_queue/ → Anomaly_Validator → validation_results/
```

