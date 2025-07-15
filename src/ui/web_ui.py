from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys
import json
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.nlp_models.sentiment_analyzer import SentimentAnalyzer
from src.nlp_models.preference_embedder import PreferenceEmbedder
from src.core_logic.ranking import rank_places

app = Flask(__name__)
CORS(app)

# Global variables for models 
embedder = None
sentiment_analyzer = None
predefined_attributes = [
    "Cozy", "Trendy", "Romantic", "Lively", "Quiet", "Elegant", "Casual", "Artistic",
    "Bohemian", "Family-Friendly", "Pet-Friendly", "Outdoor Seating", "Good for Groups",
    "Good for Solo", "Gourmet", "Comfort Food", "Healthy", "Vegan-Friendly", "Dessert",
    "Coffee", "Date", "Scenic View", "Parking Available", "Wheelchair Accessible",
    "Wi-Fi Available", "Workspace"
]

def load_places_data():
    with open('data/raw/basir_combined_places_reviews_final_20250710_142531.json', 'r', encoding='utf-8') as f:
        return json.load(f)

places_data = load_places_data()

def initialize_models():
  
    """
    Initializes the PreferenceEmbedder and SentimentAnalyzer models.

    If the jinaai/jina-embeddings-v3 embedding model  can't be loaded, it uses .
    """
    global embedder, sentiment_analyzer
    
    try:
        print("Initializing models...")
        
        embedder = PreferenceEmbedder(model_name='jinaai/jina-embeddings-v3')
        embedder.generate_attribute_embeddings(predefined_attributes)
        
        # Initialize 
        try:
            sentiment_analyzer = SentimentAnalyzer(
                en_model_path='hayn404/roberta-finetuned',
                ar_model_path='hayn404/araberta_finetuned'
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
        scores = {}
        review_lower = review_text.lower()
        
        for attr in self.attribute_names:
            attr_lower = attr.lower()
            if attr_lower in review_lower or attr_lower.replace('-', ' ') in review_lower:
                scores[attr] = random.uniform(0.6, 0.9)
            else:
                scores[attr] = random.uniform(0.0, 0.3)
                
        return scores

def haversine(coord1, coord2):
    R = 6371  # Earth radius in km
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@app.route('/')
def index():
    """Serve the main UI page."""
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """
    Endpoint to get a ranked list of places based on user's preferences.

    Accepts a JSON payload with the following keys:

    - preferences (dict): A dictionary of attribute names to their weights.
    - budget (int): Budget category (0-3).
    - city (str): City name (used to filter places by city).
    - max_distance (float): Maximum distance from user's location to filter places.
    - place_name (str): Name of a place to use as user's location (if provided).

    Returns a JSON response with the following keys:

    - success (bool): Whether the request was successful.
    - recommendations (list): A list of dictionaries containing the recommended places.
        Each dictionary contains the following keys:

        - name (str)
        - address (str)
        - budget (int)
        - scoring_details (dict): A dictionary of scoring details
    """
    try:
        data = request.get_json()
        preferences = data.get('preferences', {})
        user_budget = data.get('budget', 2)
        user_city = data.get('city', '').strip().lower()
        max_distance = float(data.get('max_distance', 50))  # default 50km if not provided
        place_name = data.get('place_name', '').strip().lower()

        total_weight = sum(preferences.values())
        if total_weight > 0:
            preferences = {k: v / total_weight for k, v in preferences.items()}

        # Filter places by city 
        filtered_places = places_data
        if user_city:
            filtered_places = [p for p in places_data if user_city in p.get('address', '').strip().lower()]
        if not filtered_places:
            return jsonify({"success": False, "error": "No places found for the specified city."}), 404

        # If place_name is provided, use its coordinates as user's location, set default to first place in city or fallback
        user_coords = None
        if place_name:
            match = next((p for p in filtered_places if p.get('name', '').strip().lower() == place_name), None)
            if match and 'latitude' in match and 'longitude' in match:
                user_coords = (match['latitude'], match['longitude'])
        if not user_coords:
        
            user_coords = (filtered_places[0].get('latitude', 30.033333), filtered_places[0].get('longitude', 31.233334))

        budget_map = {
            "low": 0,
            "average": 1,
            "mid": 2,
            "high": 2,
            "expensive": 3
        }
        # coords check
        for p in filtered_places:
            if 'coords' not in p or not p['coords']:
                lat = p.get('latitude')
                lon = p.get('longitude')
                if lat is not None and lon is not None:
                    p['coords'] = (lat, lon)
                else:
                    p['coords'] = (0.0, 0.0)
            # budget check
            if 'budget' not in p or p['budget'] is None:
                br = str(p.get('budget_range', '')).strip().lower()
                p['budget'] = budget_map.get(br, 1)  # default to 1 (average)

        # filtering by max_distance
        filtered_places = [
            p for p in filtered_places
            if 'coords' in p and haversine(user_coords, p['coords']) <= max_distance
        ]
        if not filtered_places:
            return jsonify({"success": False, "error": "No places found within the specified distance."}), 404

        user_data = {
            "preferences": preferences,
            "budget": user_budget,
            "coords": user_coords
        }

        ranked_places = rank_places(
            user_data=user_data,
            places_list=filtered_places.copy(),
            embedder=embedder,
            sentiment_analyzer=sentiment_analyzer,
            predefined_attributes=predefined_attributes,
            max_distance_km=max_distance
        )

        response = []
        for place in ranked_places:
            response.append({
                "name": place["name"],
                "address": place.get("address", "Unknown"),
                "budget": place.get("budget", place.get("avg_price_usd", 0)),
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
