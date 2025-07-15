from transformers import pipeline
from typing import List, Dict, Any

def _detect_language_by_character(text: str) -> str:
    """
    Detects if a text is predominantly Arabic or English by counting characters.
    
    This is a private helper function for the SentimentAnalyzer class.

    Args:
        text (str): The input text to analyze.

    Returns:
        str: 'ar' for Arabic, 'en' for English, or 'unknown'.
    """
    arabic_chars = 0
    english_chars = 0

    if not isinstance(text, str):
        return 'unknown'

    for char in text:
        # Check for Arabic characters within the Unicode range
        if '\u0600' <= char <= '\u06FF':
            arabic_chars += 1
        # Check for English letters (a-z, A-Z)
        elif 'a' <= char.lower() <= 'z':
            english_chars += 1
            
    if arabic_chars > english_chars:
        return 'ar'
    elif english_chars > arabic_chars:
        return 'en'
    else:
        # ambiguous
        return 'unknown'

class SentimentAnalyzer:
    """
    Manages sentiment analysis models for English and Arabic to score reviews.
    """
    def __init__(self, en_model_path: str = './models/roberta', ar_model_path: str = './models/araberta'):
        """
        Initializes and loads sentiment analysis models from specified paths.

        Args:
            en_model_path (str): The file path to the English sentiment model.
            ar_model_path (str): The file path to the Arabic sentiment model.
        """
        self.pipelines: Dict[str, Any] = {}
        try:
            print(f"Loading English sentiment model from: {en_model_path}...")
            self.pipelines['en'] = pipeline("sentiment-analysis", model=en_model_path)
            
            print(f"Loading Arabic sentiment model from: {ar_model_path}...")
            self.pipelines['ar'] = pipeline("sentiment-analysis", model=ar_model_path)
            
            print(" All sentiment models loaded successfully.")
        except Exception as e:
            print(f"Error loading sentiment models: {e}")
            self.pipelines = {} 

    def analyze_reviews(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyzes a list of review dictionaries and adds a sentiment score to each.

        The score is +1 for positive, -1 for negative, and 0 for neutral/unknown.

        Args:
            reviews (List[Dict[str, Any]]): A list of review dictionaries,
                each expected to have a 'text' key.

        Returns:
            List[Dict[str, Any]]: The same list of dictionaries, with a new
                'sentiment_score' key added to each one.
        """
        if not self.pipelines:
            print("Warning: Sentiment models not loaded. Scores will default to 0.")
            for review in reviews:
                review['sentiment_score'] = 0
            return reviews

        for review in reviews:
            text = review.get('text', '')
            
            if not text or not text.strip():
                review['sentiment_score'] = 0.5
                continue

            lang = _detect_language_by_character(text)

            if lang in self.pipelines:
                result = self.pipelines[lang](text)[0]
                # The models return LABEL_1 (positive) or LABEL_0 (negative)
                review['sentiment_score'] = 1 if result['label'] == 'LABEL_1' else 0
            else:
                # Default to a neutral score for unknown or mixed languages
                review['sentiment_score'] = 0.5
        
        return reviews