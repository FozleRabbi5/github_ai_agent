from django.contrib import admin
from .models import RepositoryIndex, QuestionHistory
# Register your models here.

admin.site.register(RepositoryIndex)
admin.site.register(QuestionHistory)
