"""
Prompt Templates for Multi-Agent System

Structured prompts following Paper 2's approach for different agent roles.
"""

from typing import Dict, Any, List


class PromptTemplates:
    """
    Collection of prompt templates for different agent roles.
    
    Based on the multi-agent framework from Park (2024).
    """
    
    @staticmethod
    def data_conversion_agent(
        anomaly_data: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Data Conversion Agent: Transform tabular anomaly data into LLM-comprehensible format.
        
        This is the first step in Paper 2's framework - converting structured data
        into natural language questions that subsequent agents can process.
        """
        return f"""You are a Data Conversion Agent specializing in transforming financial data into clear, structured questions for analysis.

**Your Task:** Convert the following anomaly detection result into a comprehensive question format that other AI agents can understand and investigate.

**Raw Anomaly Data:**
{PromptTemplates._format_dict(anomaly_data)}

**Metadata:**
{PromptTemplates._format_dict(metadata)}

**Convert this into:**
1. A clear problem statement
2. Key questions that need to be answered
3. Important context for investigation
4. Expected data points to verify

**Output Format:**
Provide a structured brief that includes:
- ANOMALY SUMMARY: One-line description
- KEY QUESTIONS: List of specific questions to answer
- CONTEXT: Background information needed
- VERIFICATION POINTS: What evidence to look for"""
    
    @staticmethod
    def web_content_summarizer(
        articles: List[Dict[str, Any]],
        anomaly_context: Dict[str, Any],
        max_summary_length: int = 15000
    ) -> str:
        """
        Web Content Summarizer: Pre-RAG LLM layer for comprehensive summarization.
        
        NEW LAYER: Before RAG, use Gemini Flash 2.5 to create a detailed summary of all
        web content that preserves important details while organizing information clearly.
        
        Flow: Raw scraped content (50k+ chars) → LLM summary (10-20k chars) → RAG (2.5k chars)
        """
        # Compile all content with metadata
        all_content = []
        for i, article in enumerate(articles):
            text = article.get('text') or article.get('snippet') or article.get('description', '')
            if not text:
                continue
            
            title = article.get('title', f'Article {i+1}')
            source = article.get('url') or article.get('source', 'Unknown')
            date = article.get('published_at') or article.get('scraped_at', 'N/A')
            
            all_content.append(f"""
**Article {i+1}: {title}**
Source: {source}
Date: {date}
Content: {text[:5000]}{"..." if len(text) > 5000 else ""}
---""")
        
        combined_content = "\n".join(all_content)
        total_chars = len(combined_content)
        
        entity = anomaly_context.get('entity_name') or anomaly_context.get('entity_id')
        amount = anomaly_context.get('amount', 0)
        gl_name = anomaly_context.get('gl_name', '')
        period = anomaly_context.get('period', '')
        
        return f"""You are an expert financial analyst specializing in summarizing web research for audit investigations.

**Investigation Context:**
- Entity: {entity}
- GL Account: {gl_name}
- Anomaly Amount: ${abs(amount):,.0f}
- Period: {period}
- Investigation Focus: Why did this unusual transaction occur?

**Web Content to Summarize ({len(articles)} sources, {total_chars:,} characters):**
{combined_content}

**Your Task:**
Create an EXTREMELY DETAILED and COMPREHENSIVE summary that will be fed into a RAG (Retrieval-Augmented Generation) system.

**Requirements:**

1. **Preserve ALL Important Details** - Don't just give high-level overview, include:
   - Specific amounts, dates, and figures mentioned
   - Names of people, companies, and entities
   - Transaction details and business events
   - Quotes and key statements from executives or officials
   - Context and background information
   - Technical details and specifications

2. **Organize by Relevance** - Structure so most relevant information comes first:
   - Direct mentions of the entity or related transactions
   - Financial events, announcements, or deals in the time period
   - Industry trends or market conditions affecting the entity
   - Regulatory or compliance matters
   - Competitor activities or market dynamics
   - General background information

3. **Maintain Source Attribution** - For each piece of information, note which article it came from

4. **Flag Important Patterns** - Identify and highlight:
   - Recurring themes across multiple sources
   - Contradictions or inconsistencies between sources
   - Timeline of events (chronological ordering)
   - Causal relationships (X led to Y)
   - Market reactions and impacts

5. **Length Target**: Aim for {max_summary_length:,} characters - be thorough, not brief!
   - This summary will be chunked by RAG, so more detail = better retrieval
   - Include context that helps understand WHY things happened

**Output Format:**
Provide a well-structured summary with clear sections. Include as much detail as possible while staying organized.

Begin your comprehensive summary:"""
    
    @staticmethod
    def web_research_agent(
        investigation_brief: str,
        entity_name: str,
        period: str
    ) -> str:
        """
        Web Research Agent: Research external sources to verify anomaly authenticity.
        
        Paper 2: "verifies the authenticity of anomalies by researching web-based resources,
        such as press releases from data publishers, major news articles or social media posts."
        """
        return f"""You are a Web Research Agent specializing in financial market verification.

**Investigation Brief:**
{investigation_brief}

**Research Parameters:**
- Entity: {entity_name}
- Time Period: {period}

**Your Task:**
Based on the retrieved web content provided below, analyze whether external sources explain or validate this anomaly.

**Focus On:**
1. Press releases or company announcements
2. Major news articles about the entity or industry
3. Market events or economic factors
4. Regulatory filings or compliance notices
5. Industry trends or sector-wide patterns

**Retrieved Web Content (via RAG):**
{{rag_context}}

**Analysis Required:**
1. Does the web content explain this anomaly?
2. Are there external events that support this transaction?
3. Is this anomaly consistent with publicly available information?
4. Any red flags or contradictions?

**Output Format (JSON):**
{{
    "explains_anomaly": true/false,
    "confidence": 0.0-1.0,
    "supporting_evidence": ["evidence1", "evidence2", ...],
    "external_events": ["event1", "event2", ...],
    "concerns": ["concern1", ...] or [],
    "summary": "brief summary of findings"
}}"""
    
    @staticmethod
    def institutional_knowledge_agent(
        investigation_brief: str,
        historical_context: str = ""
    ) -> str:
        """
        Institutional Knowledge Agent: Leverage domain expertise and historical patterns.
        
        Paper 2: "Functioning as an experienced market analyst, this agent leverages an
        extensive domain knowledge to provide context and explanations for the detected anomalies."
        """
        return f"""You are an Institutional Knowledge Agent - an experienced financial analyst with deep domain expertise.

**Investigation Brief:**
{investigation_brief}

**Historical Context:**
{historical_context or "No specific historical context provided."}

**Your Task:**
Based on your expertise in financial accounting, auditing standards, and industry practices, provide insights on this anomaly.

**Analyze:**
1. Is this transaction type normal for this GL account?
2. What are typical patterns for this entity/industry?
3. Historical precedents for similar anomalies
4. Accounting standards and compliance considerations
5. Common legitimate reasons for such transactions
6. Red flags from an auditor's perspective

**Consider:**
- Seasonal patterns and business cycles
- Industry-specific transactions
- Regulatory requirements
- Standard operating procedures
- Past similar cases and their resolutions

**Output Format (JSON):**
{{
    "assessment": "NORMAL/UNUSUAL/SUSPICIOUS",
    "explanation": "detailed analysis",
    "historical_precedents": ["precedent1", ...],
    "accounting_considerations": ["consideration1", ...],
    "risk_factors": ["risk1", ...] or [],
    "recommendation": "suggested action"
}}"""
    
    @staticmethod
    def cross_checking_agent(
        investigation_brief: str,
        comparable_data: List[Dict[str, Any]] = None
    ) -> str:
        """
        Cross-Checking Agent: Validate through cross-referencing with other sources.
        
        Paper 2: "Dedicated to validating data through cross-referencing with other reliable sources,
        this agent plays an essential role in confirming or disputing the anomalies identified."
        """
        comparable_str = PromptTemplates._format_list(comparable_data) if comparable_data else "No comparable data available"
        
        return f"""You are a Cross-Checking Agent specializing in data validation through cross-referencing.

**Investigation Brief:**
{investigation_brief}

**Comparable Data/Benchmarks:**
{comparable_str}

**Your Task:**
Cross-reference the anomaly with other reliable data sources to validate or dispute it.

**Validation Steps:**
1. Compare with peer entities (similar companies/departments)
2. Compare with historical data for same entity
3. Check against industry benchmarks
4. Verify consistency across related GL accounts
5. Validate against expected ranges and patterns

**Analysis Points:**
- Is this anomaly consistent with peer data?
- Does it align with historical patterns?
- Are related accounts showing similar trends?
- Is the magnitude within reasonable bounds?

**Output Format (JSON):**
{{
    "validation_result": "CONFIRMED/DISPUTED/INCONCLUSIVE",
    "confidence": 0.0-1.0,
    "peer_comparison": "comparison findings",
    "historical_comparison": "historical analysis",
    "related_accounts": ["account1: status", ...],
    "discrepancies": ["discrepancy1", ...] or [],
    "conclusion": "overall assessment"
}}"""
    
    @staticmethod
    def consolidation_agent(
        expert_findings: List[Dict[str, Any]]
    ) -> str:
        """
        Consolidation and Reporting Agent: Synthesize all expert analyses into unified report.
        
        Paper 2: "An agent specialised in report synthesis consolidates the insights from
        all data expert agents, crafting a summary report that highlights the key findings."
        
        IMPORTANT: System NEVER unflags anomalies. Only assigns RED_FLAG or YELLOW_FLAG priority.
        """
        findings_str = "\n\n".join([
            f"**{i+1}. {finding.get('agent_name', 'Expert')} Findings:**\n{PromptTemplates._format_dict(finding)}"
            for i, finding in enumerate(expert_findings)
        ])
        
        return f"""You are a Consolidation Agent responsible for synthesizing expert analyses into a comprehensive report.

**CRITICAL RULE:** Anomalies are NEVER unflagged. All require human review. Assign RED_FLAG or YELLOW_FLAG priority only.

**Expert Findings:**
{findings_str}

**Your Task:**
Consolidate all expert analyses into a unified report with priority assessment.

**Report Should Include:**
1. **Executive Summary**: High-level overview of findings
2. **Key Findings**: Most important discoveries from all experts
3. **Document Authenticity**: Assessment of supporting documents (authentic/suspicious/fake)
4. **Explanation Quality**: How well the anomaly is explained
5. **Fraud Indicators**: Any signs of fraudulent activity
6. **Consensus Points**: Where experts agree
7. **Divergent Views**: Where experts disagree or have concerns
8. **Priority Flag**: RED_FLAG (urgent) or YELLOW_FLAG (explained but needs review)

**Priority Flag Guidelines:**

**RED_FLAG (High Priority):**
- Documents are suspicious, fake, or missing
- No clear explanation for anomaly
- Fraud indicators present
- Contradictory evidence
- Requires urgent investigation

**YELLOW_FLAG (Lower Priority - Still Requires Review):**
- Documents are 100% authentic
- Clear, complete explanation of why anomaly occurred
- All evidence aligns
- Legitimate business reason
- Lower urgency but still needs human approval

**Output Format (JSON):**
{{
    "executive_summary": "brief overview",
    "priority_flag": "RED_FLAG/YELLOW_FLAG",
    "document_authenticity": "AUTHENTIC/SUSPICIOUS/FAKE/MISSING",
    "explanation_quality": "COMPLETE/PARTIAL/NONE",
    "key_findings": ["finding1", "finding2", ...],
    "fraud_indicators": ["indicator1", ...] or [],
    "consensus": ["point1", ...],
    "concerns": ["concern1", ...] or [],
    "evidence_summary": "consolidated evidence",
    "why_anomaly_occurred": "explanation if known",
    "requires_urgent_review": true/false,
    "full_report": "detailed comprehensive report"
}}"""
    
    @staticmethod
    def management_discussion(
        consolidated_report: str,
        roles: List[str] = None
    ) -> str:
        """
        Management Discussion: Panel of management agents debate and reach consensus.
        
        Paper 2: "These management agents are engineered to adopt high-level perspectives,
        contrasting with the detail-oriented focus of the data expert agents."
        """
        roles = roles or ['CFO', 'Audit Director', 'Compliance Officer', 'Risk Manager']
        roles_str = ', '.join(roles)
        
        return f"""You are simulating a Management Discussion Panel with the following roles: {roles_str}

**Consolidated Expert Report:**
{consolidated_report}

**Discussion Objective:**
The management team must review the expert findings and reach consensus on the appropriate course of action.

**Discussion Format:**
Each role should provide their perspective:

1. **CFO**: Financial impact, budget implications, strategic considerations
2. **Audit Director**: Audit risk, compliance, internal controls
3. **Compliance Officer**: Regulatory requirements, policy adherence, legal risks
4. **Risk Manager**: Enterprise risk, mitigation strategies, exposure assessment

**Discussion Flow:**
1. Initial reactions from each role
2. Debate on key concerns or disagreements
3. Discussion of implications and consequences
4. Consensus building on recommended action
5. Final decision and action plan

**Output Format (JSON):**
{{
    "discussion_transcript": "realistic multi-turn discussion between roles",
    "key_concerns": ["concern1", ...],
    "consensus": "APPROVE/INVESTIGATE/ESCALATE/REJECT",
    "action_plan": "detailed next steps",
    "dissenting_views": ["dissent1", ...] or [],
    "final_decision": "management's final decision with rationale"
}}"""
    
    @staticmethod
    def _format_dict(data: Dict[str, Any], indent: int = 0) -> str:
        """Format dictionary for readable display in prompts."""
        lines = []
        prefix = "  " * indent
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}- {key}:")
                lines.append(PromptTemplates._format_dict(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}- {key}: {value}")
            else:
                lines.append(f"{prefix}- {key}: {value}")
        return "\n".join(lines)
    
    @staticmethod
    def _format_list(data: List[Any]) -> str:
        """Format list for readable display in prompts."""
        if not data:
            return "No data available"
        return "\n".join([f"- {item}" for item in data])


if __name__ == "__main__":
    # Test prompt templates
    print("Testing Prompt Templates...")
    
    test_anomaly = {
        'gl_account': '101000',
        'gl_name': 'Cash',
        'entity_id': 'E001',
        'period': '2024-10',
        'amount': 5000000,
        'expected_range': '50000-500000',
        'anomaly_score': 2.5
    }
    
    test_metadata = {
        'detection_method': 'Autoencoder reconstruction error',
        'threshold': 0.15,
        'error': 0.245
    }
    
    # Test data conversion prompt
    prompt = PromptTemplates.data_conversion_agent(test_anomaly, test_metadata)
    print("\n=== Data Conversion Agent Prompt ===")
    print(prompt[:500] + "...")
    
    # Test web research prompt
    prompt = PromptTemplates.web_research_agent("Investigate unusual $5M cash transaction", "Adani Ports", "2024-10")
    print("\n=== Web Research Agent Prompt ===")
    print(prompt[:500] + "...")
    
    print("\n✓ Prompt templates working correctly!")

