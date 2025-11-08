import logging
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.conf import settings
from django.shortcuts import redirect
import json
import os
import time
import re
from core_APP.models import Conversation, Message


logger = logging.getLogger(__name__)

# MCP Session Cache
mcp_session_cache = {}


def dashboard_view(request):
    """Dashboard view."""

    if not request.user.is_authenticated:
        return redirect("landing_page")
    print("Dashboard view for User: ", request.user.id)
    if request.user.user_type == 1:
        # Admin
        return render(request, 'dashboard/dashboard.html')
    elif request.user.user_type in [2, 3]:
        # Tower Lead and Finance Controller
        return render(request, 'dashboard/dashboard.html')
    elif request.user.user_type == 4:
        # User
        return render(request, 'dashboard/dashboard.html')
    else:
        return render(request, 'dashboard/dashboard.html')
    

def get_mcp_client(user_email, conversation_id):
    """
    Get or create MCP client with caching for session management.
    """
    # Skip if no Composio key configured
    if not settings.COMPOSIO_API_KEY:
        return None
    
    session_key = f"{user_email}-{conversation_id}"
    
    if session_key in mcp_session_cache:
        logger.debug(f'Reusing cached MCP session: {session_key}')
        return mcp_session_cache[session_key]
    
    try:
        # Try to import Composio client
        try:
            from composio.core import ComposioToolSet, Action
            from composio.client.enums import ActionType
        except ImportError:
            try:
                from composio import Composio, ToolSet
            except ImportError:
                logger.warning('Composio SDK not available, skipping MCP integration')
                return None
        
        logger.info('MCP session framework ready')
        
        # Cache placeholder for future use
        mcp_session_cache[session_key] = {
            'available': True,
            'note': 'Composio integration ready when API key is configured'
        }
        
        return mcp_session_cache[session_key]
    except Exception as e:
        logger.error(f'Failed to initialize MCP client: {str(e)}')
        return None


def is_balance_sheet_query(message: str) -> bool:
    """
    Detect if this is a balance sheet assurance query.
    """
    lower_message = message.lower()
    
    specific_phrases = [
        'balance sheet assurance',
        'gl variance',
        'hygiene score',
        'trial balance validation',
        'supporting document status',
        'compliance report',
        'variance analysis',
        'gl account review',
        'adani balance',
        'balance guardian'
    ]
    
    if any(phrase in lower_message for phrase in specific_phrases):
        return True
    
    variance_pattern = r'show.*gl.*variance.*>.*\d+|variance.*>\s*\d+.*%|gl.*account.*>\s*\d+'
    if re.search(variance_pattern, lower_message):
        return True
    
    return False


def is_accounting_query(messages: list) -> bool:
    """
    Detect if this is a financial/accounting query.
    """
    query_text = ' '.join([msg.get('content', '') for msg in messages])
    return bool(re.search(
        r'general ledger|trial balance|GL|TB|debit|credit|accounting|financial|balance sheet|variance|audit|compliance',
        query_text,
        re.IGNORECASE
    ))


def is_tb_generation_request(message: str) -> bool:
    """
    Detect if this is a Trial Balance generation request.
    """
    return bool(re.search(
        r'convert.*general ledger.*trial|generate.*trial balance|create.*trial balance|GL.*to.*TB',
        message,
        re.IGNORECASE
    ))


@csrf_exempt
@require_http_methods(["POST"])
def chat_stream(request):
    """
    Enhanced streaming chat endpoint with MCP support, Gemini integration,
    and accounting-specific features.
    """
    try:
        payload = json.loads(request.body.decode('utf-8')) if request.body else {}
        messages_list = payload.get('messages') or []
    except json.JSONDecodeError:
        return HttpResponseBadRequest('Invalid JSON')

    # Auth check
    if not request.user or not request.user.is_authenticated:
        return JsonResponse({'error': 'Unauthorized - Please sign in'}, status=401)

    # Get conversation
    conversation_id = payload.get('conversationId')
    conversation_obj = None
    if conversation_id:
        try:
            conversation_obj = Conversation.objects.get(id=conversation_id, user=request.user)
        except Conversation.DoesNotExist:
            return JsonResponse({'error': 'Conversation not found'}, status=404)

    # Extract user message
    user_content = ''
    for msg in reversed(messages_list):
        if (msg.get('role') or '').lower() == 'user':
            user_content = msg.get('content') or ''
            break

    # Create conversation if first message
    is_first_message = not conversation_obj
    if is_first_message:
        title = generate_conversation_title(user_content)
        conversation_obj = Conversation.objects.create(user=request.user, title=title)

    # Save user message
    Message.objects.create(
        conversation=conversation_obj,
        user=request.user,
        content=user_content,
        role='user',
    )

    # Detect query type
    is_accounting = is_accounting_query(messages_list)
    is_balance_sheet = is_balance_sheet_query(user_content)
    is_tb_request = is_tb_generation_request(user_content)

    # Check for TB job status request
    job_status_match = re.search(r'job\s+([a-f0-9-]{36}|[a-f0-9-]{32})', user_content, re.IGNORECASE)
    
    if job_status_match and re.search(r'status.*job|job.*status', user_content, re.IGNORECASE):
        job_id = job_status_match.group(1)
        # For now, return placeholder - implement TB job system separately
        status_response = f"📊 **Trial Balance Job Status**\n\n**Job ID**: `{job_id}`\n**Status**: Under development\n\nTrial Balance job system will be implemented in a future update."
        
        Message.objects.create(
            conversation=conversation_obj,
            user=request.user,
            content=status_response,
            role='assistant',
        )
        
        response = StreamingHttpResponse([status_response], content_type='text/plain; charset=utf-8')
        response['X-Conversation-Id'] = str(conversation_obj.id)
        return response

    # Prepare system prompt
    system_prompt = """You are Nirva, an AI assistant with CPA-level accounting expertise that can interact with 500+ applications through Composio's Tool Router."""

    if is_accounting:
        system_prompt += """

🧮 **ACCOUNTING & FINANCIAL EXPERTISE MODE ACTIVATED**

You are now operating with specialized accounting knowledge. Follow these CRITICAL rules:

**FUNDAMENTAL ACCOUNTING PRINCIPLES:**

- **The Accounting Equation**: Assets = Liabilities + Equity (ALWAYS verify this holds)
- **Debit/Credit Rules**: 
  • Assets & Expenses: Debit increases, Credit decreases
  • Liabilities, Equity & Revenue: Credit increases, Debit decreases
- **Trial Balance Rule**: Total Debits MUST equal Total Credits (sum = 0)

**GENERAL LEDGER TO TRIAL BALANCE CONVERSION PROTOCOL:**

1. **NEVER ASSUME DATA STRUCTURE** - Always examine and confirm:
   - Column names and their meaning
   - Date ranges and accounting periods
   - Currency and number formats
   - Account coding system used

2. **VALIDATION REQUIREMENTS**:
   - Verify GL data completeness before processing
   - Check for duplicate entries or missing transactions
   - Ensure account codes match standard chart of accounts
   - Validate that all entries have both debit and credit components

3. **CALCULATION METHODOLOGY**:
   - Group transactions by account code/name
   - Sum debits and credits separately for each account
   - Calculate net balance (debit - credit) for each account
   - MANDATORY: Verify total debits = total credits
   - Flag any accounts with unusual balances

**WHEN PROCESSING FINANCIAL DATA**:
- State your understanding of the data structure BEFORE processing
- Show sample calculations for verification
- Highlight any accounts that don't follow normal balance patterns
- Provide detailed reconciliation between source GL and final TB
"""

    system_prompt += """
**Response Formatting Guidelines:**
- Always format your responses using Markdown syntax
- Use **bold** for emphasis and important points
- Use bullet points and numbered lists for clarity
- Format links as [text](url)
- Use code blocks with ``` for code snippets
- Use inline code with ` for commands, file names, and technical terms
- Use headings (##, ###) to organize longer responses
- Make your responses clear, concise, and well-structured

**Tool Execution Guidelines:**
- Explain what you're doing before using tools
- Provide clear feedback about the results
- Include relevant links when appropriate
"""

    # Generate AI response
    accumulated = []
    
    def generate_stream():
        # Try to use Google Gemini if key is available
        if settings.GOOGLE_AI_API_KEY:
            yield from stream_google_gemini(messages_list, system_prompt, is_accounting)
        elif settings.OPENAI_API_KEY:
            yield from stream_openai(messages_list, system_prompt)
        else:
            yield from stream_fallback(user_content)

    def yielding_and_persist():
        for chunk in generate_stream():
            accumulated.append(chunk)
            yield chunk
        
        # Save assistant message after streaming completes
        assistant_text = ''.join(accumulated)
        if assistant_text:
            Message.objects.create(
                conversation=conversation_obj,
                user=request.user,
                content=assistant_text,
                role='assistant',
            )

    response = StreamingHttpResponse(yielding_and_persist(), content_type='text/plain; charset=utf-8')
    response['X-Conversation-Id'] = str(conversation_obj.id)
    return response


def stream_google_gemini(messages_list, system_prompt, is_accounting=False):
    """
    Stream response from Google Gemini 2.5 Pro.
    """
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=settings.GOOGLE_AI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')  # or 'gemini-pro'
        
        # Prepare messages with system prompt
        gemini_messages = [{'role': 'user', 'parts': [system_prompt]}]
        for msg in messages_list:
            gemini_messages.append({
                'role': msg.get('role', 'user'),
                'parts': [msg.get('content', '')]
            })
        
        # Configure generation
        temperature = 0.1 if is_accounting else 0.3
        
        # Generate stream
        response = model.generate_content(
            gemini_messages,
            stream=True,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=8192,
            )
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        logger.error(f'Google Gemini streaming error: {str(e)}')
        yield f"\n\n*Error: Could not stream from Google Gemini. {str(e)}*"


def stream_openai(messages_list, system_prompt):
    """
    Stream response from OpenAI.
    """
    import requests
    
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {'Authorization': f'Bearer {settings.OPENAI_API_KEY}', 'Content-Type': 'application/json'}
    body = {
        'model': settings.AI_MODEL,
        'stream': True,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            *[{'role': m.get('role', 'user'), 'content': m.get('content', '')} for m in messages_list]
        ]
    }
    
    try:
        with requests.post(url, headers=headers, json=body, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith(b'data: '):
                    data = line[len(b'data: '):]
                    if data == b"[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj.get('choices', [{}])[0].get('delta', {})
                        content = delta.get('content')
                        if content:
                            yield content
                    except Exception:
                        continue
    except Exception as e:
        logger.error(f'OpenAI streaming error: {str(e)}')
        yield f"\n\n*Error: Could not stream from OpenAI. {str(e)}*"


def stream_fallback(text: str):
    """
    Fallback streaming when no API keys are configured.
    """
    fallback_message = (
        f"Here is a simulated response based on your message: {text}\n\n"
        "**Configuration Required:**\n"
        "- Set GOOGLE_AI_API_KEY for Google Gemini\n"
        "- OR set OPENAI_API_KEY for OpenAI\n"
        "- Set COMPOSIO_API_KEY for full MCP tool integration\n"
    )
    for ch in fallback_message:
        yield ch
        time.sleep(0.01)


def generate_conversation_title(first_message: str) -> str:
    """
    Generate a conversation title from the first message.
    """
    clean = (first_message or '').replace('\n', ' ').strip()
    if len(clean) <= 50:
        return clean or 'New conversation'
    words = clean.split(' ')
    title = ''
    for w in words:
        if len(title) + len(w) + (1 if title else 0) <= 47:
            title += ((' ' if title else '') + w)
        else:
            break
    return (title or clean[:47]) + '...'


@login_required
def list_conversations(request):
    """
    List all conversations for the authenticated user.
    """
    data = [
        {
            'id': str(c.id),
            'title': c.title,
            'created_at': c.created_at.isoformat(),
            'updated_at': c.updated_at.isoformat(),
        }
        for c in Conversation.objects.filter(user=request.user).order_by('-updated_at')
    ]
    return JsonResponse({'conversations': data})


@login_required
def list_messages(request, conv_id: str):
    """
    List all messages in a conversation.
    """
    try:
        conv = Conversation.objects.get(id=conv_id, user=request.user)
    except Conversation.DoesNotExist:
        return JsonResponse({'error': 'Conversation not found'}, status=404)
    
    msgs = [
        {
            'id': str(m.id),
            'conversation_id': str(conv.id),
            'content': m.content,
            'role': m.role,
            'created_at': m.created_at.isoformat(),
        }
        for m in conv.messages.all().order_by('created_at')
    ]
    return JsonResponse({'messages': msgs})
