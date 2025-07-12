from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys
import json

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.nlp_models.sentiment_analyzer import SentimentAnalyzer
from src.nlp_models.preference_embedder import PreferenceEmbedder
from src.core_logic.ranking import rank_places

app = Flask(__name__)
CORS(app)

# Global variables for models (initialize once)
embedder = None
sentiment_analyzer = None
predefined_attributes = [
    "Cozy", "Trendy", "Romantic", "Lively", "Quiet", "Elegant", "Casual", "Artistic",
    "Bohemian", "Family-Friendly", "Pet-Friendly", "Outdoor Seating", "Good for Groups",
    "Good for Solo", "Gourmet", "Comfort Food", "Healthy", "Vegan-Friendly", "Dessert",
    "Coffee", "Date", "Scenic View", "Parking Available", "Wheelchair Accessible",
    "Wi-Fi Available", "Workspace"
]

# Placeholder data for demonstration
placeholder_places = [
    {
        "name": "The Grand Cafe",
        "reviews": [
            {"text": "Very trendy spot with great music! Always lively and energetic."},
            {"text": "Perfect for groups, great atmosphere for socializing."},
            {"text": "Modern decor and excellent coffee selection."}
        ],
        "budget": 2,
        "coords": (30.0444, 31.2357),
        "address": "Downtown Cairo"
    },
    {
        "name": "Quiet Corner Books",
        "reviews": [
            {"text": "A very quiet and cozy place for reading and studying."},
            {"text": "Perfect workspace with reliable Wi-Fi and comfortable seating."},
            {"text": "Great coffee and peaceful environment for solo work."}
        ],
        "budget": 1,
        "coords": (30.0561, 31.2394),
        "address": "Zamalek District"
    },
    {
        "name": "Uptown Lounge",
        "reviews": [
            {"text": "Super energetic and trendy. The place to be seen!"},
            {"text": "Elegant setting with gourmet food and artistic decor."},
            {"text": "Perfect for romantic dates with scenic city views."}
        ],
        "budget": 3,
        "coords": (30.0450, 31.2360),
        "address": "New Cairo"
    },
    {
        "name": "Family Garden Restaurant",
        "reviews": [
            {"text": "Amazing family-friendly atmosphere with outdoor seating."},
            {"text": "Great for kids, pet-friendly with excellent comfort food."},
            {"text": "Wheelchair accessible with convenient parking."}
        ],
        "budget": 1,
        "coords": (30.0333, 31.2200),
        "address": "Maadi"
    },
    {
        "name": "Bohemian Arts Cafe",
        "reviews": [
            {"text": "Incredibly artistic and bohemian vibe with unique decor."},
            {"text": "Perfect for creative minds, great coffee and desserts."},
            {"text": "Casual atmosphere that's inspiring and relaxing."}
        ],
        "budget": 2,
        "coords": (30.0600, 31.2400),
        "address": "Heliopolis"
    },
    {
        "name": "Healthy Bites Kitchen",
        "reviews": [
            {"text": "Excellent healthy options with many vegan-friendly choices."},
            {"text": "Fresh ingredients and nutritious meals that actually taste great."},
            {"text": "Clean, bright atmosphere perfect for health-conscious diners."}
        ],
        "budget": 2,
        "coords": (30.0400, 31.2300),
        "address": "Dokki"
    }
]

def initialize_models():
    """Initialize the ML models once when the app starts."""
    global embedder, sentiment_analyzer
    
    try:
        print("Initializing models...")
        
        # Initialize PreferenceEmbedder
        embedder = PreferenceEmbedder(model_name='jinaai/jina-embeddings-v3')
        embedder.generate_attribute_embeddings(predefined_attributes)
        
        # Initialize SentimentAnalyzer with fallback
        try:
            sentiment_analyzer = SentimentAnalyzer(
                en_model_path='./models/roberta',
                ar_model_path='./models/araberta'
            )
        except Exception as e:
            print(f"Warning: Could not load sentiment models: {e}")
            print("Using mock sentiment analyzer for demo...")
            sentiment_analyzer = MockSentimentAnalyzer()
            
        print("Models initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        print("Using mock models for demo...")
        embedder = MockPreferenceEmbedder()
        sentiment_analyzer = MockSentimentAnalyzer()

class MockSentimentAnalyzer:
    """Mock sentiment analyzer for demo purposes."""
    def analyze_reviews(self, reviews):
        import random
        for review in reviews:
            review['sentiment_score'] = random.uniform(0.4, 0.9)
        return reviews

class MockPreferenceEmbedder:
    """Mock preference embedder for demo purposes."""
    def __init__(self):
        self.attribute_names = predefined_attributes
        
    def generate_attribute_embeddings(self, attributes):
        self.attribute_names = attributes
        
    def get_review_attribute_scores(self, review_text):
        import random
        # Simple keyword matching for demo
        scores = {}
        review_lower = review_text.lower()
        
        for attr in self.attribute_names:
            attr_lower = attr.lower()
            if attr_lower in review_lower or attr_lower.replace('-', ' ') in review_lower:
                scores[attr] = random.uniform(0.6, 0.9)
            else:
                scores[attr] = random.uniform(0.0, 0.3)
                
        return scores

@app.route('/')
def index():
    """Serve the main UI page."""
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """API endpoint to get place recommendations."""
    try:
        data = request.get_json()
        
        # Extract user preferences
        preferences = data.get('preferences', {})
        user_budget = data.get('budget', 2)
        user_coords = data.get('coords', [29.9595, 31.2589])  # Default to Maadi
        
        # Normalize preferences
        total_weight = sum(preferences.values())
        if total_weight > 0:
            preferences = {k: v / total_weight for k, v in preferences.items()}
        
        # Prepare user data
        user_data = {
            "preferences": preferences,
            "budget": user_budget,
            "coords": tuple(user_coords)
        }
        
        # Get recommendations using the actual ranking logic
        ranked_places = rank_places(
            user_data=user_data,
            places_list=placeholder_places.copy(),
            embedder=embedder,
            sentiment_analyzer=sentiment_analyzer,
            predefined_attributes=predefined_attributes
        )
        
        # Format response
        response = []
        for place in ranked_places:
            response.append({
                "name": place["name"],
                "address": place.get("address", "Unknown"),
                "budget": place["budget"],
                "scoring_details": place["scoring_details"]
            })
        
        return jsonify({"success": True, "recommendations": response})
        
    except Exception as e:
        print(f"Error in recommendation endpoint: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/attributes')
def get_attributes():
    """API endpoint to get predefined attributes."""
    return jsonify({"attributes": predefined_attributes})

if __name__ == '__main__':
    initialize_models()
    app.run(debug=True, port=5000)
