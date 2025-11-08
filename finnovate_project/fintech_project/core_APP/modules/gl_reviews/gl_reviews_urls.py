from django.urls import path

from .gl_reviews import gl_reviews_view, balance_sheet_view, upload_gl_supporting_document, submit_gl_review_preparer


urlpatterns = [
    path('', gl_reviews_view, name='gl_reviews_page'),
    path('tier3/', balance_sheet_view, name='reports_tier3'),
    path('upload-document/', upload_gl_supporting_document, name='upload_gl_supporting_document'),
    path('submit-review/', submit_gl_review_preparer, name='submit_gl_review_preparer'),
]

