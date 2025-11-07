# Priority Flag System Documentation

## Overview

The Anomaly Validator implements a **two-tier flag system** where **all anomalies require human review**. The system cannot unflag anomalies, but assigns priority levels for review queue management.

## Flag Types

### 🔴 RED FLAG (High Priority - Urgent Review Required)

**Assigned When:**
- No supporting documents provided
- Documents are suspicious, fake, or tampered
- Documents don't explain the anomaly
- Fraud indicators detected
- Contradicts institutional knowledge or external context
- Missing critical information
- Unusual patterns without explanation
- Document authenticity check fails

**Action Required:**
- **Immediate investigation** by senior auditor
- Forensic document analysis if fraud suspected
- Cross-reference with original sources
- Interview stakeholders
- Escalate to management if necessary

**Review Priority:** Within 24-48 hours

---

### 🟡 YELLOW FLAG (Lower Priority - Explained but Needs Review)

**Assigned When:**
- Documents are **100% authentic** (passed fraud detection)
- Documents are **highly relevant** to the transaction (relevance score ≥ 7/10)
- Evidence in documents **fully explains** why anomaly occurred
- All evidence aligns consistently across multiple sources
- Legitimate business reason can be constructed from evidence:
  - One-time transaction (acquisition, divestment, etc.)
  - Policy change or process update
  - Seasonal variation with documentation
  - Known event confirmed by web research
- Web context supports the evidence (if available)
- Institutional knowledge consistent with findings

**What "Explains the Anomaly" Means:**

The system analyzes **what the documents show** (not what they explicitly say):

1. **Evidence Matching:**
   - Invoice amount = GL entry amount
   - Bank statement confirms transaction occurred
   - Contract terms align with transaction details
   - Dates are consistent across documents

2. **Business Logic:**
   - Transaction makes sense given the evidence
   - Scale appropriate for business size
   - Timing logical (not suspicious)
   - Vendor/counterparty verifiable

3. **Completeness:**
   - All questions answered by evidence
   - No gaps in the transaction story
   - Supporting documents form complete picture

**Examples:**

**Example 1: Large Cash Inflow**
- Anomaly: GL 101000 (Cash) +$5M (unusual)
- Documents Provided:
  - Bank statement showing $5M wire transfer from "ABC Corp"
  - Sale agreement for subsidiary sale to ABC Corp
  - Board resolution approving sale
- Web Context: Reuters article confirms ABC Corp acquired subsidiary
- **Analysis:** Documents show legitimate sale → Wire transfer amount matches → Board approved → Publicly announced
- **Decision:** YELLOW FLAG ✓

**Example 2: Unusual Expense**
- Anomaly: GL 600000 (Consulting) +$2M (10x normal)
- Documents Provided:
  - Consulting invoice for $2M from XYZ Consultants
  - SOW (Statement of Work) for special project
  - Purchase order matching invoice
  - Payment confirmation
- Web Context: Company announced major restructuring project
- **Analysis:** Documents show legitimate consulting engagement → Amount matches across all docs → Aligns with announced project
- **Decision:** YELLOW FLAG ✓

**Example 3: Recurring Transaction Pattern**
- Anomaly: GL 200000 (Inventory) unusual spike
- Documents Provided:
  - Multiple purchase orders from regular supplier
  - Receiving reports confirming delivery
  - Invoices totaling GL amount
- Analysis: Regular supplier → Quantities match orders → Inventory buildup for new product launch (from web research)
- **Decision:** YELLOW FLAG ✓

**Action Required:**
- **Standard review** by auditor (not urgent)
- Verify explanation aligns with business objectives
- Confirm all documentation is properly filed
- Sign off on exception

**Review Priority:** Within 5-10 business days

---

## Document Authenticity Check

All supporting documents undergo fraud detection analysis:

### Fraud Detection Checklist

1. **Formatting Issues**
   - Font inconsistencies
   - Poor quality or pixelation
   - Misaligned elements
   - Unusual spacing

2. **Content Inconsistencies**
   - Illogical dates
   - Suspiciously round/odd amounts
   - Missing standard elements
   - Internal contradictions

3. **Metadata Red Flags**
   - Creation date doesn't match document date
   - Recent file creation for old documents
   - Multiple rapid edits

4. **Business Logic Issues**
   - Unverifiable vendors/parties
   - Unusual terms or conditions
   - Scale doesn't match business size

5. **Language/Style Issues**
   - Unusual wording for official documents
   - AI-generated appearance
   - Generic lacking details

### Authenticity Ratings

- **AUTHENTIC**: No concerning signs, passes all checks
- **SUSPICIOUS**: Some red flags, needs investigation
- **FAKE**: Multiple serious indicators of forgery

---

## Workflow

```
Anomaly Detected
    ↓
Document Collection (invoices, bank statements, contracts, etc.)
    ↓
STEP 1: Document Authenticity Check
    ↓
    ├─→ FAKE/SUSPICIOUS → RED FLAG (Stop here)
    │
    └─→ AUTHENTIC (Documents are real)
            ↓
        STEP 2: Document Relevance Check
            ↓
            ├─→ NOT RELEVANT (Wrong GL, wrong period, unrelated) → RED FLAG
            │
            └─→ RELEVANT (Documents relate to this transaction)
                    ↓
                STEP 3: Evidence Analysis
                What do the documents show?
                - Invoice amount matches GL entry?
                - Bank statement confirms payment?
                - Contract explains unusual terms?
                - Board approval for one-time expense?
                    ↓
                STEP 4: Explanation Construction
                (AI analyzes evidence + web context)
                "Can we explain WHY this anomaly occurred?"
                    ↓
                    ├─→ NO: Evidence insufficient/contradictory → RED FLAG
                    │   Examples:
                    │   - Invoice present but amount doesn't match
                    │   - Documents missing key information
                    │   - Evidence contradicts web research
                    │
                    └─→ YES: Evidence fully explains anomaly → YELLOW FLAG
                        Examples:
                        - Invoice $5M + Bank statement $5M + Press release about acquisition
                        - Contract shows one-time payment + Board resolution approving it
                        - Multiple invoices totaling GL amount + Vendor verification via web
                            ↓
                        Human Review (Lower Priority)
                            ↓
                        Approval/Rejection
```

---

## Key Principles

### 1. **Never Unflag**
- System cannot mark anomaly as "no action needed"
- All anomalies reach human auditors
- Prevents AI from missing critical fraud

### 2. **Fraud Detection First**
- Document authenticity checked before explanation analysis
- Fake documents automatically trigger RED FLAG
- Multiple layers of fraud detection

### 3. **Explanation Quality Matters**
- YELLOW FLAG requires **complete** explanation
- Partial explanations still get RED FLAG
- Must be supported by authentic documents

### 4. **Human-in-the-Loop**
- AI assists, humans decide
- Final approval always requires auditor sign-off
- System provides prioritization, not elimination

---

## Metrics & Monitoring

Track these metrics for system effectiveness:

1. **Flag Distribution**
   - % RED FLAGS vs YELLOW FLAGS
   - Should stabilize around 70% RED / 30% YELLOW for healthy system

2. **False Positive Rate**
   - % YELLOW FLAGS rejected after human review
   - Target: < 5%

3. **Document Fraud Detection**
   - % documents flagged as FAKE/SUSPICIOUS
   - % confirmed fake after investigation

4. **Review Time**
   - Average time to resolve RED FLAGS (target: 24-48h)
   - Average time to resolve YELLOW FLAGS (target: 5-10 days)

---

## Configuration

Edit `config/validator_config.yaml`:

```yaml
validation:
  # Never set below 7.0 for YELLOW FLAG eligibility
  document_authenticity_threshold: 7.0
  
  # Explanation quality threshold for YELLOW FLAG
  explanation_completeness_threshold: 0.9
  
  # Confidence required for YELLOW FLAG
  yellow_flag_confidence: 0.8
  
  # Fraud indicator count that triggers RED FLAG
  max_fraud_indicators_for_yellow: 0
```

---

## Examples

### Example 1: RED FLAG
```
Anomaly: GL 101000 (Cash) - $10M spike
Documents: Bank statement (creation date: today, document date: last month)
Authenticity: FAKE (metadata inconsistency)
Decision: RED FLAG - Fraudulent document detected
```

### Example 2: YELLOW FLAG (Well-Documented Transaction)
```
Anomaly: GL 101000 (Cash) - $5M spike
Documents Provided: 
  - Bank statement showing $5M wire transfer from "BuyerCo"
  - Asset sale agreement between company and BuyerCo
  - Board resolution dated 2 weeks prior approving sale
  - Press release announcing transaction
  
Document Analysis:
  - Authenticity: AUTHENTIC (all documents pass fraud detection)
  - Relevance: 9/10 (directly related to transaction)
  - Evidence: Wire amount ($5M) matches sale agreement matches board resolution
  - Timing: Logical sequence (board approval → sale → payment)
  
Web Context: Reuters article confirms BuyerCo acquired subsidiary for $5M
  
AI Explanation Construction:
  "The $5M cash spike occurred due to a subsidiary sale to BuyerCo. 
   Bank statement confirms wire transfer, sale agreement specifies $5M price,
   board pre-approved transaction, and public announcement validates event.
   All evidence aligns."
  
Decision: YELLOW FLAG ✓
Reason: Documents provide complete explanation - authentic, relevant, consistent
```

### Example 3: RED FLAG (Documents Don't Explain)
```
Anomaly: GL 500000 (Consulting Expense) - $2M unusual charge
Documents Provided: 
  - Invoice from "ABC Consulting" for $2M
  
Document Analysis:
  - Authenticity: AUTHENTIC (invoice passes fraud detection)
  - Relevance: 6/10 (relates to GL account but...)
  - Evidence Gap: No SOW, no contract, no approval documentation
  - Business Logic: No record of ABC Consulting engagement in company records
  
Web Context: ABC Consulting not found in web search, no company website
  
AI Analysis:
  "While invoice is authentic, there is no supporting evidence for why this
   engagement occurred. No contract, no statement of work, no approval chain.
   Vendor cannot be verified externally. Transaction lacks business justification."
  
Decision: RED FLAG ⚠️
Reason: Authentic document but insufficient evidence to explain anomaly
```

### Example 4: RED FLAG (Evidence Contradicts)
```
Anomaly: GL 400000 (Revenue) - $3M unusual credit
Documents Provided:
  - Customer invoice for $3M
  - Sales contract
  
Document Analysis:
  - Authenticity: AUTHENTIC
  - Relevance: 8/10
  - Evidence Contradiction: Invoice date is 3 months AFTER revenue recorded
  - Business Logic Issue: Revenue recognized before sale completed
  
Decision: RED FLAG ⚠️
Reason: Documents are real but evidence shows improper revenue recognition
```

---

## Integration with Human Review

### RED FLAG Workflow
1. Alert sent to senior auditor immediately
2. Full document package provided
3. Fraud analysis report included
4. Escalation path defined
5. Resolution tracking mandatory

### YELLOW FLAG Workflow
1. Added to standard review queue
2. Documents pre-validated as authentic
3. Explanation provided for quick review
4. Simple approve/reject decision
5. Optional: Batch review multiple YELLOW FLAGS

---

Last Updated: October 30, 2025
Version: 1.0

