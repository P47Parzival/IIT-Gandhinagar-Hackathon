# Document Analysis Guide

## Understanding Documents vs. Explanations

### Key Concept

**Documents don't "explain" - they provide EVIDENCE.**  
**The AI constructs an EXPLANATION from analyzing the evidence.**

---

## Document Types and What They Show

### 1. **Invoices**
**What they show:**
- Amount charged
- Vendor/supplier name
- Description of goods/services
- Date of transaction
- Payment terms
- Invoice number (for tracking)

**How AI uses them:**
```
Invoice shows: $50,000 consulting fee from XYZ Corp
GL Entry shows: $50,000 debit to Consulting Expense
AI Reasoning: "Amount matches. Is XYZ Corp legitimate? Yes (web search confirms). 
               Is consulting expense expected? Checking institutional knowledge..."
```

### 2. **Bank Statements**
**What they show:**
- Actual money movement
- Transaction dates
- Counterparty names
- Account balances
- Wire transfer details

**How AI uses them:**
```
Bank statement shows: $5M wire received from BuyerCo on Oct 15
GL Entry shows: $5M credit to Cash on Oct 15
AI Reasoning: "Amount and date match perfectly. BuyerCo is verified entity. 
               This confirms actual cash receipt."
```

### 3. **Contracts/Agreements**
**What they show:**
- Terms and conditions
- Agreed amounts
- Payment schedules
- Parties involved
- Signatures/approvals
- Effective dates

**How AI uses them:**
```
Contract shows: Annual license fee of $1M, payable quarterly ($250K)
GL Entry shows: $250K debit to Software Expense
AI Reasoning: "This matches contract terms. Is this the correct quarter? Yes. 
               Contract is signed and dated before transaction."
```

### 4. **Purchase Orders**
**What they show:**
- Items/services ordered
- Quantities
- Unit prices
- Delivery terms
- PO number (audit trail)

**How AI uses them:**
```
PO shows: 1000 units @ $500 = $500K
Invoice shows: $500K
GL Entry shows: $500K debit to Inventory
AI Reasoning: "Three-way match: PO → Invoice → GL. Quantities and amounts align."
```

### 5. **Board Resolutions/Approvals**
**What they show:**
- Authorization for transactions
- Approval dates
- Voting records
- Conditions/restrictions

**How AI uses them:**
```
Board resolution shows: Approved subsidiary sale for up to $6M on Sept 1
Bank statement shows: $5M received Oct 15
AI Reasoning: "Transaction authorized before execution. Amount within approved limit."
```

### 6. **Receiving Reports/Delivery Confirmations**
**What they show:**
- Goods actually received
- Received dates
- Quantities received
- Condition of goods

**How AI uses them:**
```
Receiving report shows: 1000 units received Oct 10
Invoice shows: 1000 units billed Oct 12
GL Entry shows: Inventory increase Oct 12
AI Reasoning: "Physical receipt confirmed before accounting entry. Proper sequence."
```

---

## Analysis Process: From Documents to Explanation

### Step 1: Document Authentication
**Question:** Are these documents real or fake?

**Checks:**
- Metadata analysis (creation dates, edit history)
- Formatting consistency
- Business logic (does vendor exist?)
- Content contradictions

**Output:** AUTHENTIC / SUSPICIOUS / FAKE

---

### Step 2: Document Relevance
**Question:** Do these documents relate to this specific transaction?

**Checks:**
- GL account matches document type
- Period matches transaction date
- Entity matches document parties
- Amount in reasonable range

**Output:** Relevance score 0-10

---

### Step 3: Evidence Extraction
**Question:** What facts can we extract from documents?

**Extracted:**
```python
evidence = {
    "transaction_amount": 5000000,
    "counterparty": "BuyerCo Inc",
    "transaction_date": "2024-10-15",
    "transaction_type": "Asset Sale",
    "authorization": "Board Resolution 2024-09-30",
    "payment_method": "Wire Transfer",
    "supporting_docs": ["bank_statement", "sale_agreement", "board_resolution"]
}
```

---

### Step 4: Cross-Verification
**Question:** Does evidence from different documents align?

**Example:**
```
Bank Statement:    $5M from BuyerCo on Oct 15
Sale Agreement:    $5M purchase price, closing Oct 15
Board Resolution:  Approved sale up to $6M
Press Release:     Announced Oct 16 (day after)
GL Entry:          $5M cash credit on Oct 15
```

**AI Analysis:** ✓ All sources confirm same facts → High confidence

---

### Step 5: Business Logic Check
**Question:** Does this make business sense?

**Checks:**
- Transaction scale appropriate for entity size
- Counterparty is real, verifiable entity
- Timing is logical (not suspicious)
- Terms are standard for this type of transaction
- No red flags in institutional knowledge

---

### Step 6: Explanation Construction
**Question:** Can we construct a complete explanation from evidence?

**Good Explanation (YELLOW FLAG):**
```
"The $5M cash spike in GL 101000 occurred due to the sale of the Manufacturing 
Division subsidiary to BuyerCo Inc. 

Evidence Analysis:
- Bank statement confirms $5M wire received from BuyerCo on Oct 15, 2024
- Asset sale agreement specifies $5M purchase price with Oct 15 closing
- Board resolution dated Sept 30 pre-approved sale up to $6M
- Reuters press release Oct 16 publicly announced the transaction
- All amounts, dates, and parties align across documents

Business Logic: Transaction is legitimate, properly authorized, properly documented,
and publicly disclosed. The unusual cash spike is a one-time event with clear cause."

Decision: YELLOW FLAG (still needs human review, but well-explained)
```

**Incomplete Explanation (RED FLAG):**
```
"The $2M consulting expense in GL 500000 is supported by an invoice from ABC 
Consulting.

Evidence Analysis:
- Invoice is authentic (passed fraud detection)
- Invoice amount matches GL entry
- However: No contract, no SOW, no PO, no approval documentation
- ABC Consulting cannot be verified online (no website, no business records)
- Institutional knowledge has no record of this engagement

Business Logic: While the invoice is real, there is insufficient evidence to 
explain why this engagement occurred or whether it was authorized."

Decision: RED FLAG (requires urgent investigation)
```

---

## Document Completeness Checklist

### For YELLOW FLAG, need:

**Transaction Evidence:**
- [ ] Primary document (invoice, contract, bank statement)
- [ ] Amount verification (cross-check across sources)
- [ ] Date verification (logical timing)
- [ ] Party verification (counterparty is real)

**Authorization Evidence:**
- [ ] Approval chain documented (PO, board resolution, etc.)
- [ ] Within authorized limits
- [ ] Proper signatories

**External Verification:**
- [ ] Counterparty verifiable online
- [ ] Transaction consistent with public information
- [ ] No contradictions in web research

**Business Logic:**
- [ ] Transaction makes business sense
- [ ] Scale appropriate
- [ ] Timing not suspicious
- [ ] Consistent with institutional knowledge

---

## Red Flags That Prevent YELLOW FLAG

Even with authentic documents:

1. **Incomplete Evidence**
   - Missing key supporting documents
   - Gaps in transaction trail
   - Cannot verify counterparty

2. **Contradictory Evidence**
   - Amounts don't match across documents
   - Dates are illogical (payment before contract)
   - Parties don't align

3. **No Authorization**
   - No approval documentation
   - Exceeds authorized limits
   - Wrong approvers

4. **Business Logic Failure**
   - Transaction doesn't make sense
   - Counterparty suspicious/unverifiable
   - Timing is suspicious

5. **External Contradictions**
   - Web research contradicts documents
   - Public records show different information
   - Institutional knowledge conflicts

---

## Examples of Evidence-Based Analysis

### Example: Complete Evidence Chain ✓

```
Anomaly: $800K inventory purchase (2x normal)

Documents:
1. Purchase Order #12345 - 1000 units @ $800 each = $800K
2. Supplier Invoice #98765 - $800K
3. Receiving Report - 1000 units received, inspected, accepted
4. Payment confirmation - $800K paid via ACH
5. Warehouse receipt - Units in stock

Cross-Verification:
- PO → Invoice → Receiving → Payment all match $800K
- Dates logical: PO (Oct 1) → Delivery (Oct 10) → Invoice (Oct 12) → Payment (Oct 30)
- Supplier is verified, 5-year relationship
- Units physically in inventory

Business Logic:
- New product launch requires inventory buildup (from board minutes)
- Supplier capacity confirmed (from web research)
- Unit price consistent with historical pricing

AI Explanation:
"The $800K inventory spike is a planned inventory buildup for the new product 
launch scheduled for November. All documentation aligns: purchase order, receiving 
confirmation, and physical inventory receipt are verified. This is a normal business 
transaction with complete audit trail."

Decision: YELLOW FLAG ✓
```

### Example: Authentic but Insufficient Evidence ⚠️

```
Anomaly: $500K marketing expense (new vendor)

Documents:
1. Invoice from "Digital Marketing Pro" for $500K

Cross-Verification:
- Only one document
- No contract, no SOW, no approval
- Cannot find "Digital Marketing Pro" online
- No previous relationship with this vendor

Business Logic Issues:
- Large payment to new vendor without contract
- No competitive bidding documentation
- No deliverables specified
- Vendor cannot be verified

AI Explanation:
"While the invoice is authentic, there is insufficient evidence to explain this 
transaction. The vendor is unverifiable, there is no contract or statement of work, 
and no approval chain. This requires investigation."

Decision: RED FLAG ⚠️
```

---

## Summary

**Documents = Evidence (raw facts)**  
**AI Analysis = Processing evidence**  
**Explanation = Constructed narrative from evidence**

**YELLOW FLAG requires:**
- Authentic documents
- Complete evidence
- Logical business story
- No contradictions
- External verification

**If any piece missing → RED FLAG**

---

Last Updated: October 30, 2025

