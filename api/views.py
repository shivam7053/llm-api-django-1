from django.http import JsonResponse
from transformers import pipeline

# Load the question-answering pipeline
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

def answer_question(request):
    question = request.GET.get('question', '')
    context = request.GET.get('context', '')
    if not question or not context:
        return JsonResponse({'error': 'Both question and context parameters are required.'}, status=400)
    result = qa_pipeline(question=question, context=context)
    return JsonResponse({'answer': result['answer']})
