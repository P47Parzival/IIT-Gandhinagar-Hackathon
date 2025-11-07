"""
Gemini Flash Lite Client for LLM-based Agent Operations

Provides a unified interface for all agent interactions with Google's Gemini API.
"""

import os
import time
from typing import Dict, List, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class GeminiClient:
    """
    Client for interacting with Google Gemini Flash Lite API.
    
    This client provides methods for various agent tasks:
    - Document relevance assessment
    - Anomaly analysis and validation
    - Natural language explanation generation
    - Multi-agent discussions
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 30
    ):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (or uses GEMINI_API_KEY env var)
            model: Model name (default: gemini-1.5-flash for speed)
            temperature: Sampling temperature (0.0-1.0, lower = more deterministic)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        )
        
        print(f"✓ Gemini client initialized: {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generate response from Gemini with retry logic.
        
        Args:
            prompt: User prompt
            system_instruction: Optional system instruction for role-playing
            
        Returns:
            Generated text response
        """
        try:
            # Create model with system instruction if provided
            if system_instruction:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_instruction,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    )
                )
            else:
                model = self.model
            
            response = model.generate_content(
                prompt,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                }
            )
            
            # Check if response was blocked by safety filters
            if not response.candidates or not response.candidates[0].content.parts:
                # Response was blocked - check safety ratings
                if response.candidates and hasattr(response.candidates[0], 'safety_ratings'):
                    blocked_reasons = [
                        f"{rating.category.name}: {rating.probability.name}"
                        for rating in response.candidates[0].safety_ratings
                        if rating.probability.name != "NEGLIGIBLE"
                    ]
                    error_msg = f"Content blocked by safety filters: {', '.join(blocked_reasons)}"
                else:
                    error_msg = "Content blocked by safety filters (no details available)"
                
                print(f"⚠️  {error_msg}")
                # Return a safe default response instead of crashing
                return '{"error": "safety_block", "message": "Response blocked by content safety filters"}'
            
            return response.text
            
        except (KeyError, AttributeError) as e:
            # Safety filters often raise KeyError or AttributeError with category names
            # This happens when trying to access response.text or response.candidates
            error_str = str(e).strip("'\"")
            if 'dangerous' in error_str.lower() or 'harm' in error_str.lower():
                print(f"⚠️  Content safety block detected: {error_str}")
            else:
                print(f"⚠️  Gemini API KeyError/AttributeError: {error_str}")
            return '{"error": "safety_block", "message": "Response blocked by content safety filters"}'
            
        except Exception as e:
            error_str = str(e)
            print(f"⚠️  Gemini API error: {error_str[:200]}")
            
            # Handle specific error cases
            if any(word in error_str.lower() for word in ['safety', 'blocked', 'dangerous', 'harm_category', 'finish_reason']):
                return '{"error": "safety_block", "message": "Response blocked by content safety filters"}'
            elif "quota" in error_str.lower() or "429" in error_str:
                return '{"error": "quota_exceeded", "message": "API quota exceeded"}'
            
            # For unexpected errors, return error JSON instead of crashing
            return f'{{"error": "api_error", "message": "API call failed: {error_str[:100]}"}}'
    
    def assess_document_relevance(
        self,
        document_text: str,
        anomaly_context: Dict[str, Any],
        max_doc_length: int = 5000
    ) -> Dict[str, Any]:
        """
        Assess if a document is relevant to the detected anomaly.
        
        Based on Paper 2's approach of verifying authenticity through documents.
        
        Args:
            document_text: Extracted text from supporting document
            anomaly_context: Dictionary with anomaly details
            max_doc_length: Maximum document length to process
            
        Returns:
            Dictionary with:
                - relevance_score: 0-10 score
                - is_relevant: Boolean (score >= 7)
                - reasoning: Explanation of relevance
                - key_findings: Important points from document
        """
        # Truncate long documents
        if len(document_text) > max_doc_length:
            document_text = document_text[:max_doc_length] + "...[truncated]"
        
        system_instruction = (
            "You are an expert financial auditor specializing in GL account validation. "
            "Your task is to assess if supporting documents are relevant and support "
            "the reported GL account balances."
        )
        
        prompt = f"""Analyze the following supporting document for a GL account that was flagged as anomalous.

**Anomaly Details:**
- GL Account: {anomaly_context.get('gl_account', 'Unknown')}
- Entity: {anomaly_context.get('entity_id', 'Unknown')}
- Period: {anomaly_context.get('period', 'Unknown')}
- Amount: {anomaly_context.get('amount', 'Unknown')}
- Anomaly Type: {anomaly_context.get('anomaly_type', 'High reconstruction error')}

**Supporting Document:**
{document_text}

**Assessment Task:**
1. Rate the document's relevance to this GL account (0-10 scale)
   - 10 = Perfectly relevant, directly explains the transaction
   - 7-9 = Relevant, provides supporting evidence
   - 4-6 = Partially relevant, some connection
   - 0-3 = Not relevant, unrelated to this account

2. Explain your reasoning

3. Extract key findings that support or contradict the GL balance

**Output Format (JSON):**
{{
    "relevance_score": <0-10>,
    "is_relevant": <true/false>,
    "reasoning": "<explanation>",
    "key_findings": ["<finding1>", "<finding2>", ...]
}}"""

        response = self.generate(prompt, system_instruction)
        
        # Parse JSON response
        try:
            import json
            # Extract JSON from response (handle markdown code blocks)
            json_text = response
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(json_text)
            return result
        except:
            # Fallback parsing
            return {
                "relevance_score": 5.0,
                "is_relevant": False,
                "reasoning": response,
                "key_findings": []
            }
    
    def analyze_anomaly(
        self,
        anomaly_data: Dict[str, Any],
        document_analysis: List[Dict[str, Any]],
        web_context: Optional[Dict[str, Any]] = None,
        institutional_knowledge: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze anomaly using all available information.
        
        This implements the Paper 2 expert agents' consolidated analysis.
        
        IMPORTANT: Anomalies are NEVER unflagged. All require human review.
        System assigns priority flags:
        - RED FLAG: High priority - suspicious, no explanation, or fake documents
        - YELLOW FLAG: Lower priority - legitimate explanation with authentic documents
        
        Args:
            anomaly_data: Detected anomaly details
            document_analysis: Results from document relevance assessment
            web_context: External context from web research
            institutional_knowledge: Domain knowledge and policies
            
        Returns:
            Dictionary with:
                - decision: 'RED_FLAG' or 'YELLOW_FLAG' (never unflagged)
                - confidence: 0-1 confidence score
                - reasoning: Detailed explanation
                - evidence: Supporting evidence for decision
                - document_authenticity: Assessment of document legitimacy
        """
        system_instruction = (
            "You are a senior financial auditor with expertise in accounting standards, "
            "financial analysis, and fraud detection. Your role is to assess anomaly priority. "
            "CRITICAL: You can NEVER unflag anomalies. All anomalies require human review. "
            "Your job is to assign RED FLAG (urgent review) or YELLOW FLAG (explained but still needs review)."
        )
        
        # Build context from all sources
        doc_summary = "\n".join([
            f"- Document {i+1}: Relevance {d.get('relevance_score', 0)}/10 - {d.get('reasoning', 'N/A')}"
            for i, d in enumerate(document_analysis)
        ])
        
        web_summary = "No web research available."
        if web_context:
            web_summary = f"""
**Web Research Findings:**
- News Articles: {len(web_context.get('news', []))} found
- Company Announcements: {len(web_context.get('company', []))} found
- Summary: {web_context.get('summary', 'N/A')}
"""
        
        knowledge_summary = institutional_knowledge or "No institutional knowledge provided."
        
        prompt = f"""Assess the priority level of this detected GL account anomaly. ALL ANOMALIES REQUIRE HUMAN REVIEW - you cannot unflag them.

**Anomaly Details:**
- GL Account: {anomaly_data.get('gl_account', 'Unknown')}
- Description: {anomaly_data.get('gl_name', 'Unknown')}
- Entity: {anomaly_data.get('entity_id', 'Unknown')}
- Period: {anomaly_data.get('period', 'Unknown')}
- Amount: ${anomaly_data.get('amount', 0):,.2f}
- Expected Range: ${anomaly_data.get('expected_min', 0):,.2f} - ${anomaly_data.get('expected_max', 0):,.2f}
- Anomaly Score: {anomaly_data.get('anomaly_score', 0):.2f} (reconstruction error / threshold)

**Document Analysis:**
{doc_summary}

{web_summary}

**Institutional Knowledge:**
{knowledge_summary}

**Decision Task:**
Assign a priority flag (you CANNOT unflag - all anomalies need human review):

1. **RED FLAG** (High Priority - Urgent Review Required):
   - No supporting documents or documents are suspicious/fake
   - Documents don't explain the anomaly
   - Contradicts institutional knowledge or web context
   - Signs of potential fraud or tampering
   - Missing critical information
   - Unusual patterns that can't be explained

2. **YELLOW FLAG** (Lower Priority - Explained but Needs Review):
   - Documents are 100% authentic and relevant
   - Clear explanation for why anomaly occurred
   - Consistent with web context and institutional knowledge
   - Legitimate business reason (e.g., one-time transaction, acquisition, policy change)
   - All supporting evidence aligns
   - **Still requires human approval but less urgent**

**Document Authenticity Check:**
Assess if documents show signs of being fake, tampered, or fraudulent:
- Inconsistent formatting
- Suspicious timing
- Missing standard elements
- Contradictory information
- Too perfect/manufactured appearance

**Output Format (JSON):**
{{
    "decision": "RED_FLAG" or "YELLOW_FLAG",
    "confidence": <0.0-1.0>,
    "reasoning": "<detailed explanation of why this anomaly occurred>",
    "evidence": ["<evidence1>", "<evidence2>", ...],
    "document_authenticity": "AUTHENTIC/SUSPICIOUS/FAKE/NO_DOCUMENTS",
    "fraud_indicators": ["<indicator1>", ...] or [],
    "explanation_quality": "COMPLETE/PARTIAL/NONE",
    "requires_urgent_review": true/false
}}"""

        response = self.generate(prompt, system_instruction)
        
        # Parse JSON response
        try:
            import json
            json_text = response
            
            # Check if response contains an error
            if response.startswith('{"error":'):
                result = json.loads(response)
                error_type = result.get("error", "unknown")
                error_msg = result.get("message", "Unknown error")
                
                return {
                    "decision": "RED_FLAG",  # Default to high priority on error
                    "confidence": 0.3,
                    "reasoning": f"Analysis failed due to {error_type}: {error_msg}. Flagging for manual review.",
                    "evidence": [],
                    "document_authenticity": "ANALYSIS_ERROR",
                    "fraud_indicators": [],
                    "explanation_quality": "NONE",
                    "requires_urgent_review": True
                }
            
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(json_text)
            # Ensure decision is never "UNFLAG"
            if result.get("decision") not in ["RED_FLAG", "YELLOW_FLAG"]:
                result["decision"] = "RED_FLAG"  # Default to high priority
            return result
        except:
            return {
                "decision": "RED_FLAG",  # Default to high priority on error
                "confidence": 0.5,
                "reasoning": response if len(response) < 500 else response[:500] + "...",
                "evidence": [],
                "document_authenticity": "UNKNOWN",
                "fraud_indicators": [],
                "explanation_quality": "NONE",
                "requires_urgent_review": True
            }
    
    def generate_explanation(
        self,
        anomaly_data: Dict[str, Any],
        validation_result: Dict[str, Any],
        document_summaries: List[str]
    ) -> Dict[str, Any]:
        """
        Generate natural language explanation for the validation result.
        
        This implements the Paper 2 Report Consolidation Agent.
        
        Args:
            anomaly_data: Original anomaly details
            validation_result: Analysis decision and reasoning
            document_summaries: Brief summaries of supporting documents
            
        Returns:
            Dictionary with:
                - summary: One-line summary
                - full_explanation: Detailed NLP explanation
                - document_summary: Combined document summary
                - recommendation: Next steps
        """
        system_instruction = (
            "You are an expert report writer for financial audits. "
            "Generate clear, concise explanations that non-technical stakeholders can understand."
        )
        
        docs_text = "\n".join([f"- {s}" for s in document_summaries]) if document_summaries else "No documents"
        
        prompt = f"""Generate a comprehensive report for the following GL account anomaly validation.

**Anomaly:**
- GL {anomaly_data.get('gl_account')}: {anomaly_data.get('gl_name')}
- Entity {anomaly_data.get('entity_id')}, Period {anomaly_data.get('period')}
- Amount: ${anomaly_data.get('amount', 0):,.2f}
- Flagged as: {anomaly_data.get('anomaly_type', 'Unusual transaction')}

**Validation Decision:** {validation_result.get('decision')}
**Confidence:** {validation_result.get('confidence', 0):.1%}
**Risk Level:** {validation_result.get('risk_level', 'UNKNOWN')}

**Analysis:** {validation_result.get('reasoning')}

**Supporting Documents:**
{docs_text}

**Generate:**
1. **One-line Summary**: Brief description of the finding (max 100 chars)
2. **Full Explanation**: Clear, professional explanation of:
   - What the anomaly is
   - Why it was flagged
   - What the validation found
   - Why the decision was made
3. **Document Summary**: Consolidated summary of all supporting documents
4. **Recommendation**: What action should be taken next

**Output Format (JSON):**
{{
    "summary": "<one-line summary>",
    "full_explanation": "<detailed explanation>",
    "document_summary": "<combined document summary>",
    "recommendation": "<next steps>"
}}"""

        response = self.generate(prompt, system_instruction)
        
        # Parse JSON response
        try:
            import json
            json_text = response
            
            # Check if response contains an error
            if response.startswith('{"error":'):
                result = json.loads(response)
                error_msg = result.get("message", "Unknown error")
                
                return {
                    "summary": "Report generation failed",
                    "full_explanation": f"Could not generate detailed explanation: {error_msg}. Anomaly requires manual review.",
                    "document_summary": "Not available due to generation error",
                    "recommendation": "Manual review required due to API limitations"
                }
            
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(json_text)
            return result
        except:
            return {
                "summary": "Validation complete",
                "full_explanation": response if len(response) < 1000 else response[:1000] + "...",
                "document_summary": "See full report",
                "recommendation": "Review findings"
            }
    
    def summarize_web_content(
        self,
        articles: List[Dict[str, Any]],
        anomaly_context: Dict[str, Any],
        max_summary_length: int = 15000
    ) -> str:
        """
        Generate comprehensive summary of all scraped web content.
        
        This is a pre-RAG summarization layer that condenses large amounts of
        web content (50k+ chars) into a focused summary (10-20k chars) containing
        the most important details relevant to the anomaly.
        
        The summary is then fed into RAG for final chunk retrieval.
        
        Args:
            articles: List of scraped articles/pages with text content
            anomaly_context: Context about the anomaly being investigated
            max_summary_length: Target length for the summary
            
        Returns:
            Comprehensive summary string containing key information
        """
        # Compile all content
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
        
        if not all_content:
            return "No web content available to summarize."
        
        combined_content = "\n".join(all_content)
        total_chars = len(combined_content)
        
        print(f"  🤖 Sending {total_chars:,} chars from {len(articles)} articles to Gemini Flash 2.5 for summarization...")
        
        system_instruction = (
            "You are an expert financial analyst specializing in summarizing web research "
            "for audit investigations. Create comprehensive, detailed summaries that preserve "
            "all important information while organizing it clearly."
        )
        
        entity = anomaly_context.get('entity_name') or anomaly_context.get('entity_id')
        amount = anomaly_context.get('amount', 0)
        gl_name = anomaly_context.get('gl_name', '')
        period = anomaly_context.get('period', '')
        
        prompt = f"""Analyze and summarize the following web research content for an anomaly investigation.

**Investigation Context:**
- Entity: {entity}
- GL Account: {gl_name}
- Anomaly Amount: ${abs(amount):,.0f}
- Period: {period}
- Investigation Focus: Why did this unusual transaction occur?

**Web Content to Summarize ({len(articles)} sources, {total_chars:,} characters):**
{combined_content}

**Your Task:**
Create an EXTREMELY DETAILED and COMPREHENSIVE summary that:

1. **Preserves ALL Important Details**: Don't just give high-level overview - include:
   - Specific amounts, dates, and figures mentioned
   - Names of people, companies, and entities
   - Transaction details and business events
   - Quotes and key statements
   - Context and background information

2. **Organize by Relevance**: Structure the summary so most relevant information comes first:
   - Direct mentions of the entity or related transactions
   - Financial events, announcements, or deals in the time period
   - Industry trends or market conditions
   - Regulatory or compliance matters
   - General background information

3. **Maintain Source Attribution**: For each piece of information, note which article it came from

4. **Flag Important Patterns**: Identify:
   - Recurring themes across multiple sources
   - Contradictions or inconsistencies
   - Timeline of events
   - Causal relationships

5. **Length Target**: Aim for {max_summary_length:,} characters - be thorough, not brief!

**Output Format:**
Provide a well-structured summary with clear sections. Include as much detail as possible while staying organized.

Begin your comprehensive summary:"""

        try:
            # Use higher token limit for summary generation
            original_max_tokens = self.max_tokens
            self.max_tokens = 8192  # Allow longer summaries
            
            # Create temporary model with higher token limit
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Slightly higher for more comprehensive coverage
                    max_output_tokens=8192,
                )
            )
            
            response = model.generate_content(
                prompt,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                }
            )
            
            # Check if response was blocked
            if not response.candidates or not response.candidates[0].content.parts:
                print(f"  ⚠️  Content blocked by safety filters")
                print(f"  ℹ️  Falling back to truncated original content")
                # Restore settings before fallback
                self.max_tokens = original_max_tokens
                return combined_content[:max_summary_length]
            
            summary = response.text
            
            # Restore original settings
            self.max_tokens = original_max_tokens
            
            print(f"  ✓ Generated summary: {len(summary):,} chars (compressed from {total_chars:,} chars)")
            print(f"  📊 Compression ratio: {(len(summary)/total_chars)*100:.1f}% of original")
            
            return summary
            
        except Exception as e:
            error_str = str(e)
            print(f"  ⚠️  Summarization failed: {error_str[:200]}")
            print(f"  ℹ️  Falling back to truncated original content")
            # Restore settings before fallback
            self.max_tokens = original_max_tokens
            # Fallback: return truncated content
            return combined_content[:max_summary_length]
    
    def management_discussion(
        self,
        expert_report: str,
        roles: List[str] = None
    ) -> Dict[str, Any]:
        """
        Simulate management panel discussion on expert findings.
        
        This implements Paper 2's Management Discussion Panel where multiple
        management agents debate from different strategic perspectives.
        
        Args:
            expert_report: Consolidated expert analysis report
            roles: List of management roles (e.g., ['CFO', 'Audit Director', 'Compliance Officer'])
            
        Returns:
            Dictionary with:
                - consensus: Agreed-upon decision
                - discussion: Transcript of discussion
                - concerns: Any raised concerns
                - final_recommendation: Action plan
        """
        if roles is None:
            roles = ['CFO', 'Audit Director', 'Compliance Officer']
        
        system_instruction = (
            "You are a panel of senior executives reviewing an audit finding. "
            "Engage in a realistic discussion, considering different perspectives, "
            "and reach a consensus on the appropriate action."
        )
        
        prompt = f"""Simulate a management discussion about the following audit finding.

**Expert Report:**
{expert_report}

**Management Panel:**
{', '.join(roles)}

**Discussion Task:**
Each management role should:
1. Review the expert findings from their perspective
2. Raise any concerns or questions
3. Discuss implications for the organization
4. Debate the recommended action
5. Reach a consensus

**Output Format (JSON):**
{{
    "consensus": "APPROVE/INVESTIGATE/ESCALATE",
    "discussion": "<realistic multi-turn discussion>",
    "concerns": ["<concern1>", "<concern2>", ...],
    "final_recommendation": "<action plan>"
}}"""

        response = self.generate(prompt, system_instruction)
        
        # Parse JSON response
        try:
            import json
            json_text = response
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(json_text)
            return result
        except:
            return {
                "consensus": "INVESTIGATE",
                "discussion": response,
                "concerns": [],
                "final_recommendation": "Further review required"
            }


if __name__ == "__main__":
    # Test the Gemini client
    print("Testing Gemini Client...")
    
    try:
        client = GeminiClient()
        
        # Test basic generation
        response = client.generate("Say 'Gemini client is working!'")
        print(f"\n✓ Basic test: {response}")
        
        # Test document relevance
        test_doc = "Bank statement showing wire transfer of $5,000,000 from investment sale dated Oct 15, 2024"
        test_anomaly = {
            "gl_account": "101000",
            "entity_id": "E001",
            "period": "2024-10",
            "amount": 5000000,
            "anomaly_type": "Unusual large debit"
        }
        
        relevance = client.assess_document_relevance(test_doc, test_anomaly)
        print(f"\n✓ Document relevance test:")
        print(f"  Score: {relevance.get('relevance_score')}/10")
        print(f"  Relevant: {relevance.get('is_relevant')}")
        
        print("\n✓ All Gemini client tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("Make sure GEMINI_API_KEY environment variable is set")

