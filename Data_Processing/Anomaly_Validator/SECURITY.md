# Security & Audit Controls

## Core Security Principles

### 1. **No Automatic Unflagging**
```python
# HARDCODED SECURITY CONTROL
CAN_UNFLAG = False  # Never set to True
```

**Why:**
- AI cannot be trusted to determine "no review needed"
- Financial fraud detection requires 100% human oversight
- False negatives (missed fraud) are more costly than false positives

**Enforcement:**
- Hardcoded in `gemini_client.py:analyze_anomaly()`
- Config setting ignored if set to True
- All LLM prompts explicitly forbid unflagging
- Response parsing validates decision is RED_FLAG or YELLOW_FLAG only

### 2. **Document Authenticity Mandatory**

All documents undergo **forensic analysis** before consideration:

```python
fraud_checks = [
    "formatting_consistency",
    "metadata_validation",  
    "business_logic_verification",
    "content_contradiction_detection",
    "language_style_analysis"
]
```

**Fraud Detection Triggers:**
- Creation date mismatch with document date
- Multiple rapid file edits
- Suspicious formatting/fonts
- Unverifiable parties/vendors
- Missing standard document elements
- AI-generated content detection

**Action on Fake Documents:**
- Automatic RED FLAG assignment
- Alert senior auditor immediately
- Preserve document for forensic investigation
- Flag entity for increased scrutiny

### 3. **Yellow Flag Strict Requirements**

YELLOW FLAG is **NOT** a free pass - it's lower priority review.

**Requirements (ALL must be met):**
```yaml
✓ Documents 100% authentic (confidence > 0.8)
✓ Explanation completeness > 90%
✓ Zero fraud indicators
✓ All evidence aligns
✓ Business reason verified
✓ Web context supports (if available)
✓ Overall confidence > 0.8
```

**If ANY requirement fails:** → RED FLAG

### 4. **Audit Trail**

Every decision is logged with:
```json
{
  "anomaly_id": "...",
  "decision": "RED_FLAG/YELLOW_FLAG",
  "timestamp": "ISO 8601",
  "documents_analyzed": ["doc1", "doc2"],
  "authenticity_scores": [0.95, 0.88],
  "fraud_indicators": [],
  "llm_reasoning": "...",
  "human_reviewer": null,  // Filled on human review
  "final_decision": null,  // Approved/Rejected
  "review_timestamp": null
}
```

**Audit Trail Never Deleted:**
- All validations stored permanently
- Enables post-incident investigation
- Tracks AI performance over time
- Regulatory compliance evidence

### 5. **Human-in-the-Loop Mandatory**

```
AI Analysis → Priority Flag → Human Review → Final Decision
     ↓              ↓               ↓              ↓
  Assist        Prioritize       Decide        Approve
```

**Human Authority:**
- Can override any AI decision
- Can escalate YELLOW to RED
- Cannot skip review (no "auto-approve")
- Must provide rejection reason

### 6. **Role-Based Access Control**

**Junior Auditor:**
- Review YELLOW FLAGS only
- Can flag for escalation
- Cannot approve RED FLAGS

**Senior Auditor:**
- Review RED FLAGS
- Can approve/reject YELLOW FLAGS
- Escalate to management if needed

**Audit Manager:**
- Review escalated cases
- Override decisions
- Access all audit trails
- Configure system parameters

### 7. **Document Chain of Custody**

```
Upload → Virus Scan → Hash → Store → Parse → Analyze → Archive
   ↓         ↓          ↓      ↓       ↓        ↓         ↓
Verify   Malware    SHA256  Encrypt  Extract  Flag    Permanent
```

**Controls:**
- SHA-256 hash on upload (detect tampering)
- Virus/malware scanning
- Encrypted storage (AES-256)
- Access logs (who viewed/downloaded)
- Tamper-evident archival

### 8. **Rate Limiting & Abuse Prevention**

**LLM API Calls:**
- Max 100 requests/minute per user
- Max 1000 requests/hour globally
- Exponential backoff on errors

**Document Uploads:**
- Max 10 documents per anomaly
- Max 50MB per document
- Max 100 uploads per user per day

**Prevents:**
- API quota exhaustion
- Cost overruns
- System abuse/DOS

### 9. **Data Privacy**

**Never Sent to LLM:**
- Personally Identifiable Information (PII)
- Social Security Numbers
- Bank account details (full numbers)
- Passwords or credentials

**Redaction Applied:**
- Account numbers masked: XXXX-XXXX-1234
- Names replaced with: [ENTITY_NAME]
- Email addresses: [REDACTED_EMAIL]

**Gemini API:**
- No training on user data (Google policy)
- Data not retained by Google
- API calls over TLS 1.3

### 10. **Incident Response Plan**

**If Fake Document Detected:**
1. Immediate RED FLAG assignment
2. Alert to senior auditor (email + SMS)
3. Lock entity for additional scrutiny
4. Preserve all related documents
5. Escalate to fraud investigation team
6. Notify management within 24 hours

**If System Compromise Detected:**
1. Disable automated processing
2. Revert to manual review
3. Audit all recent decisions
4. Investigate attack vector
5. Patch vulnerabilities
6. Resume with enhanced monitoring

### 11. **Model Monitoring**

Track AI performance to detect drift/manipulation:

```python
metrics = {
    "flag_distribution": "70% RED, 30% YELLOW (expected)",
    "fraud_detection_rate": "Track confirmed fakes",
    "false_positive_rate": "< 5% YELLOW FLAGS rejected",
    "review_time": "RED: 24-48h, YELLOW: 5-10 days",
    "human_override_rate": "Track disagreements"
}
```

**Alerts Triggered If:**
- YELLOW FLAG rate > 50% (too lenient)
- YELLOW FLAG rate < 10% (too strict)
- Human override rate > 20% (AI not reliable)
- Fraud detection rate drops suddenly

### 12. **Compliance**

**Regulatory Alignment:**
- ✓ SOX (Sarbanes-Oxley) - Audit trail requirements
- ✓ ISA (International Standards on Auditing) - Professional skepticism
- ✓ GDPR - Data privacy and retention
- ✓ SOC 2 - Security controls and monitoring

**Audit-Ready:**
- Complete decision logs
- Human review records
- System configuration history
- Performance metrics

---

## Configuration Security

**File:** `config/validator_config.yaml`

**Protected Settings (Cannot Be Overridden):**
```yaml
validation:
  can_unflag: false  # HARDCODED - ignored if changed
  max_fraud_indicators_for_yellow: 0  # Zero tolerance
  require_documents: true  # Always required
```

**Tunable Settings (With Limits):**
```yaml
validation:
  relevance_threshold: 7.0  # Min: 6.0, Max: 10.0
  authenticity_confidence_threshold: 0.8  # Min: 0.7, Max: 1.0
  yellow_flag_confidence_threshold: 0.8  # Min: 0.7, Max: 1.0
```

**Changing Thresholds:**
- Requires Audit Manager approval
- Logged in audit trail
- Applied to future validations only
- Existing decisions not retroactively changed

---

## Emergency Procedures

### Disable AI Validation
```python
# In validator_config.yaml
pipeline:
  mode: manual  # Disable automated validation
```

### Force All RED FLAGS
```python
# In validator_config.yaml
validation:
  force_red_flag_mode: true  # Emergency override
```

### Audit Recent Decisions
```bash
python scripts/audit_recent_decisions.py --days 7 --flag YELLOW
```

---

## Security Checklist for Deployment

- [ ] `CAN_UNFLAG` hardcoded to `False`
- [ ] Document authenticity checker enabled
- [ ] Fraud detection thresholds configured
- [ ] Audit trail storage configured
- [ ] Role-based access control implemented
- [ ] Document encryption enabled
- [ ] API rate limiting active
- [ ] PII redaction enabled
- [ ] Monitoring dashboards configured
- [ ] Alert rules configured
- [ ] Incident response plan documented
- [ ] Staff training completed

---

**Last Security Audit:** October 30, 2025  
**Next Review Due:** January 30, 2026  
**Security Officer:** [To Be Assigned]

