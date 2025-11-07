from django.urls import path
from .dashboard import (
    dashboard_view,
    chat_stream,
    list_conversations,
    list_messages,
)

urlpatterns = [
    path('', dashboard_view, name='dashboard_page'),
    path('api/chat', chat_stream, name='dashboard_chat_stream'),
    path('api/conversations', list_conversations, name='dashboard_conversations'),
    path('api/conversations/<uuid:conv_id>/messages', list_messages, name='dashboard_conversation_messages'),
]
