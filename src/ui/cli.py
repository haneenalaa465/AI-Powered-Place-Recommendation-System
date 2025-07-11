from src.nlp_models.sentiment_analyzer import SentimentAnalyzer
from src.nlp_models.preference_embedder import PreferenceEmbedder
from src.core_logic.ranking import rank_places

import os

# Get the absolute path of the directory containing this script (src/ui)
_UI_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up to the project's root directory
PROJECT_ROOT = os.path.abspath(os.path.join(_UI_DIR, '..'))
# Define the absolute path to the models directory
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

if __name__ == "__main__":

    # 1. Initialize the PreferenceEmbedder and generate attribute embeddings
    # Define the expanded list of predefined attributes
    predefined_attributes = [
        "Cozy", "Trendy", "Romantic", "Lively", "Quiet", "Elegant", "Casual", "artistic",
        "Bohemian", "Family-Friendly", "Pet-Friendly", "Outdoor Seating", "Good for Groups",
        "Good for Solo", "Gourmet", "Comfort Food", "Healthy", "Vegan-Friendly", "Dessert",
        "Coffee", "Date", "Scenic View", "Parking Available", "Wheelchair Accessible",
        "Wi-Fi Available", "Workspace"
    ]
    embedder = PreferenceEmbedder(model_name='facebook/mbart-large-cc25') # Changed model to mBART
    embedder.generate_attribute_embeddings(predefined_attributes)
    
    sentiment_analyzer = SentimentAnalyzer(
        en_model_path=os.path.join(MODELS_DIR, 'roberta'),
        ar_model_path=os.path.join(MODELS_DIR, 'araberta')
    )
    
    embedder.generate_attribute_embeddings(predefined_attributes)

    # --- 2. Define User Data (as before) ---
    user_data = {
        "preferences": {"Lively": 0.6, "Trendy": 0.4},
        "budget": 2, # User prefers mid-range to expensive
        "coords": (30.033333, 31.233334) # User's location (Cairo)
    }

    # --- 3. Define a LIST of places to rank ---
    places_to_rank = [
        {
            "name": "The Grand Cafe",
            "reviews": [{'text': 'Very trendy spot with great music! Always lively.'}],
            "budget": 2,
            "coords": (30.0444, 31.2357) # Close by
        },
        {
            "name": "Quiet Corner Books",
            "reviews": [{'text': 'A very quiet and cozy place for reading.'}],
            "budget": 0,
            "coords": (30.0561, 31.2394) # A bit further
        },
        {
            "name": "Uptown Lounge",
            "reviews": [{'text': 'Super energetic and trendy. The place to be seen.'}],
            "budget": 3,
            "coords": (30.0450, 31.2360) # Also close by
        }
    ]

    # --- 4. Get the ranked list of places! ---
    ranked_results = rank_places(
        user_data=user_data,
        places_list=places_to_rank,
        embedder=embedder,
        sentiment_analyzer=sentiment_analyzer,
        predefined_attributes=predefined_attributes
    )

    # --- 5. Display the sorted results ---
    print("\n--- Top Recommendations ---")
    for i, place in enumerate(ranked_results):
        score = place['scoring_details']['final_score']
        print(f"{i+1}. {place['name']} (Score: {score:.4f})")
        # Optional: print more details
        # print(f"   - Details: {place['scoring_details']}")