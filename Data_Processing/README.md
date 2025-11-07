# BalanceGuard AI - Automated Balance Sheet Assurance

**AI-powered anomaly detection and validation for financial auditing across 1000+ legal entities.**

## Overview

BalanceGuard AI combines two cutting-edge research papers to provide end-to-end automated balance sheet assurance:

1. **Paper 1: Federated Continual Learning for Anomaly Detection** (Schreyer et al., 2022)
   - Detects anomalies in GL transactions using deep autoencoder
   - Supports federated learning across multiple entities
   - Continual learning for adapting to new patterns

2. **Paper 2: LLM-based Multi-Agent Framework for Validation** (Park, 2024)
   - Validates detected anomalies using specialized AI agents
   - Analyzes supporting documents and web context
   - Generates natural language explanations
   - Assigns priority flags (RED/YELLOW) for human review

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     BalanceGuard AI                          │
└─────────────────────────────────────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                  │
   ┌────────▼─────────┐            ┌──────────▼──────────┐
   │  Paper 1 (FCL)   │            │  Paper 2 (LLM MAS)  │
   │ Anomaly Detector │───────────▶│  Anomaly Validator  │
   └──────────────────┘            └─────────────────────┘
            │                                  │
            │                       ┌──────────┴──────────┐
            │                       │                     │
   ┌────────▼─────────┐    ┌───────▼───────┐   ┌────────▼────────┐
   │ GL Transactions  │    │ Document Agent│   │ Research Agent  │
   │ (SAP/Non-SAP)   │    │ (PDF/Excel/   │   │ (Web Scraping)  │
   └──────────────────┘    │  Image OCR)   │   └─────────────────┘
                           └───────┬────────┘
                                   │
                          ┌────────▼─────────┐
                          │ Reasoning Agent  │
                          │ (Flag Assignment)│
                          └────────┬─────────┘
                                   │
                          ┌────────▼─────────┐
                          │  Report Agent    │
                          │ (NLP Explanations)│
                          └──────────────────┘
```

## Key Features

### ✅ Anomaly Detection (Paper 1)
- **Deep Autoencoder**: Detects unusual GL transactions
- **Federated Learning**: Train across multiple entities without sharing raw data
- **Continual Learning**: Adapts to new accounting periods and patterns
- **High Precision**: Low false positive rate with proper threshold tuning

### ✅ Anomaly Validation (Paper 2)
- **Multi-Agent System**: Specialized agents for documents, research, reasoning, and reporting
- **Document Analysis**: Parses PDF, Excel, CSV, and images (OCR)
- **Document Authenticity**: AI-driven fraud detection on supporting documents
- **Web Research**: Scrapes news, company sites, and regulatory filings for context
- **RAG Integration**: Efficient semantic search over large web-scraped content
- **LLM Reasoning**: Gemini Flash Lite for analysis and explanation generation
- **Two-Tier Flagging**: RED_FLAG (urgent) vs YELLOW_FLAG (standard review)
- **NEVER Unflags**: All anomalies require human review (security policy)

### ✅ Production-Ready Features
- **Real-time Processing**: Anomaly queue with priority-based processing
- **Batch Processing**: Validate large batches of anomalies
- **Comprehensive Reporting**: Natural language explanations and recommendations
- **Performance Tracking**: Detailed statistics and monitoring
- **Persistence**: Queue state and results saved to disk
- **Error Handling**: Automatic retry mechanism for failed validations

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Gemini API key (for LLM functionality)

### Setup

```bash
# Clone repository
git clone <your-repo-url>
cd IIT-Gandhinagar-Hackathon/Data_Processing

# Install Anomaly Detector dependencies
cd Anomaly_Detector
pip install -r requirements.txt

# Install Anomaly Validator dependencies
cd ../Anomaly_Validator
pip install -r requirements.txt

# Set up Gemini API key
export GEMINI_API_KEY='your-api-key-here'
```

### Optional: Install Tesseract for OCR
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## Quick Start

### 1. Train Anomaly Detector (Paper 1)

```bash
cd Anomaly_Detector

# Train on your GL data
python scripts/train_federated_continual.py \
    --data data/raw/your_gl_data.csv \
    --entities ENTITY_A ENTITY_B ENTITY_C \
    --output-dir data/models \
    --epochs 50
```

This produces:
- `data/models/federated_model.pth` (trained autoencoder)
- `data/models/preprocessor.pkl` (feature preprocessor)

### 2. Run Complete Pipeline

```bash
cd ..  # Back to Data_Processing root

# Run detection + validation pipeline
python run_full_pipeline.py \
    --data Anomaly_Detector/data/raw/ledger_fy25_qtr4.csv \
    --gemini-api-key $GEMINI_API_KEY \
    --model Anomaly_Detector/data/models/federated_oklahoma_dec2025.pth \
    --preprocessor Anomaly_Detector/data/models/preprocessor_oklahoma.pkl \
    --entity "Adani Power Mumbai" \
    --period "2025-Q4" \
    --threshold 0.1 \
    --mode batch \
    --output-dir data/pipeline_results
```

### 3. View Results

Results are saved in `data/pipeline_results/`:
- `validation_summary_report.txt` - Executive summary
- `anomalies/*.json` - Individual anomaly records
- `validation_results/*.json` - Detailed validation results

## Usage Examples

### Example 1: Quick Demo

```bash
cd Anomaly_Validator

# Run demo validation on sample anomaly
python scripts/demo_validation.py
```

This validates a single demo anomaly and shows the complete multi-agent workflow.

### Example 2: Validate Single Anomaly

```python
from Anomaly_Validator.src.pipeline import ValidatorPipeline

# Initialize pipeline
pipeline = ValidatorPipeline(
    config_path='Anomaly_Validator/config/validator_config.yaml',
    gemini_api_key='your-api-key'
)

# Create anomaly record
anomaly = {
    'anomaly_id': 'ANO_001',
    'entity_name': 'Adani Power',
    'gl_account': '6000',
    'gl_name': 'Travel Expenses',
    'amount': 125000.00,
    'expected_max': 50000.00,
    'anomaly_score': 8.5,
    # ... more fields
}

# Validate
result = pipeline.validate_single(anomaly)

print(f"Decision: {result['decision']}")
print(f"Summary: {result['summary']}")
print(f"Explanation: {result['full_explanation']}")
```

### Example 3: Real-time Processing

```python
from Anomaly_Validator.src.pipeline import ValidatorPipeline

pipeline = ValidatorPipeline(gemini_api_key='your-api-key')

# Start real-time processing thread
pipeline.start_realtime_processing()

# Submit anomalies as they're detected
for anomaly in detected_anomalies:
    anomaly_id = pipeline.submit_anomaly(anomaly)
    print(f"Submitted: {anomaly_id}")

# Check status
status = pipeline.get_status(anomaly_id)
if status['status'] == 'validated':
    print(f"Result: {status['validation_result']}")

# Stop when done
pipeline.stop_realtime_processing()
```

## Configuration

### Anomaly Detector Config

Edit `Anomaly_Detector/config/` for training parameters:
- Model architecture (shallow/deep)
- Federated learning settings
- Continual learning strategies (EWC, LwF, Replay)
- Hyperparameters

### Anomaly Validator Config

Edit `Anomaly_Validator/config/validator_config.yaml`:

```yaml
llm:
  model: gemini-1.5-flash
  temperature: 0.2
  max_tokens: 1024

validation:
  can_unflag: false  # NEVER change this
  relevance_threshold: 7.0
  authenticity_confidence_threshold: 0.8
  yellow_flag_confidence_threshold: 0.8
  max_fraud_indicators_for_yellow: 0

web_scraping:
  enable_news: true
  enable_company: true
  enable_regulatory: true
  max_articles_per_source: 5
```

## Testing

### Run Component Tests

```bash
cd Anomaly_Validator

# Test all components
python scripts/test_components.py

# Expected output:
# ✓ PASS  Gemini Client
# ✓ PASS  Document Parsers
# ✓ PASS  Anomaly Queue
# ...
```

### Run End-to-End Test

```bash
cd Anomaly_Validator

# Full validation test with real API calls
export GEMINI_API_KEY='your-api-key'
python scripts/demo_validation.py
```

## Project Structure

```
Data_Processing/
├── README.md                    # This file
├── run_full_pipeline.py         # Main integration script
│
├── Anomaly_Detector/            # Paper 1: FCL Anomaly Detection
│   ├── CONTEXT.md              # Implementation details
│   ├── src/
│   │   ├── models/             # Autoencoder models
│   │   ├── continual/          # Continual learning strategies
│   │   ├── federated/          # Federated learning
│   │   └── data/               # Preprocessing + exporter
│   ├── scripts/                # Training scripts
│   └── data/
│       ├── raw/                # Input GL data
│       ├── processed/          # Preprocessed data
│       └── models/             # Trained models
│
└── Anomaly_Validator/          # Paper 2: LLM Multi-Agent Validation
    ├── README.md
    ├── SECURITY.md             # Security policies
    ├── IMPLEMENTATION_STATUS.md
    ├── config/
    │   ├── validator_config.yaml
    │   ├── flag_system.md      # RED/YELLOW flag system
    │   └── document_analysis_guide.md
    ├── src/
    │   ├── agents/             # Multi-agent system
    │   │   ├── base_agent.py
    │   │   ├── document_agent.py
    │   │   ├── research_agent.py
    │   │   ├── reasoning_agent.py
    │   │   ├── report_agent.py
    │   │   └── coordinator.py
    │   ├── document_processing/ # Document parsers
    │   ├── web_scraping/       # Web scrapers
    │   ├── llm/                # Gemini client + RAG
    │   └── pipeline/           # Queue + pipeline
    ├── scripts/
    │   ├── test_components.py  # Component tests
    │   └── demo_validation.py  # Demo script
    └── data/
        ├── documents/          # Supporting documents
        ├── validation_results/ # Results
        └── web_cache/          # Web scraping cache
```

## Security & Compliance

### Critical Security Rules

1. **NEVER Unflags Anomalies**: All anomalies require human review
2. **Two-Tier Flag System**:
   - 🔴 **RED_FLAG**: Urgent review (fraud indicators, missing docs, high severity)
   - 🟡 **YELLOW_FLAG**: Standard review (explained by documents + web context)
3. **Document Authenticity**: AI checks for fake/tampered documents
4. **Audit Trail**: All decisions logged with full reasoning
5. **Data Privacy**: Federated learning keeps entity data decentralized

See `Anomaly_Validator/SECURITY.md` for details.

## Performance

### Typical Performance (on RTX 4070)

**Anomaly Detection (Paper 1):**
- Training: ~5-10 min per entity (10K transactions)
- Inference: <1ms per transaction
- Throughput: 100K+ transactions/second

**Anomaly Validation (Paper 2):**
- Single validation: 30-60 seconds
- Bottleneck: LLM API calls (Gemini Flash Lite)
- Throughput: ~60-120 validations/hour

**Optimization Tips:**
- Use batch mode for offline processing
- Enable web scraping cache (24h TTL)
- Use RAG to reduce LLM token usage
- Run multiple validation workers for parallelization

## Limitations & Future Work

### Current Limitations
- Web scraping limited to public sources (rate limits apply)
- OCR quality depends on document quality
- LLM costs for high volumes
- Manual review still required (by design)

### Future Enhancements
- [ ] Add latent space document similarity (replace/augment LLM relevance check)
- [ ] Implement active learning from human feedback
- [ ] Add support for non-English documents
- [ ] Build interactive dashboard (Streamlit/Plotly)
- [ ] Add email notifications for RED_FLAG anomalies
- [ ] Integrate with SAP directly for data extraction
- [ ] Add user management and responsibility matrix

## Research Papers

### Paper 1: Schreyer et al. (2022)
**Federated Continual Learning to Detect Accounting Anomalies in Financial Auditing**
- [arXiv:2210.15051](https://arxiv.org/abs/2210.15051)
- Deep autoencoder for anomaly detection
- Federated learning across entities
- Continual learning for adapting to new data

### Paper 2: Park (2024)
**Enhancing Anomaly Detection in Financial Markets with an LLM-based Multi-Agent Framework**
- [arXiv:2403.19735](https://arxiv.org/abs/2403.19735)
- Multi-agent system for validation
- LLM-powered reasoning and explanation
- Web scraping and RAG for external context

## Contributing

Contributions welcome! Areas for improvement:
- Additional document parsers
- More web scraping sources
- Alternative LLM providers
- Performance optimizations
- Dashboard and visualization

## License

[Your License]

## Acknowledgments

- Schreyer et al. for the FCL anomaly detection framework
- Park for the multi-agent validation architecture
- Adani Group for the problem statement and use case

## Support

For questions or issues:
- Check `CONTEXT.md` files for implementation details
- Review `SECURITY.md` for security policies
- Run test scripts to verify installation
- Open an issue on GitHub

---

**Built for IIT Gandhinagar Hackathon - Problem Statement #2: Automated Balance Sheet Assurance**

