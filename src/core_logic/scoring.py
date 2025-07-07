from collections import defaultdict
from typing import List, Dict, Any
from src.nlp_models.preference_embedder import PreferenceEmbedder

def calculate_place_attribute_profile(
    place_reviews: List[Dict[str, Any]],
    preference_embedder_instance: PreferenceEmbedder,
    predefined_attributes: List[str]
) -> Dict[str, float]:
    """
    Aggregates attribute scores for all reviews of a given place to create
    an overall attribute profile for that place.

    Args:
        place_reviews (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                              represents a review for a specific place.
                                              Expected format: [{'text': 'review text', ...}, ...]
        preference_embedder_instance (PreferenceEmbedder): An initialized instance of
                                                           PreferenceEmbedder with attributes
                                                           already embedded.
        predefined_attributes (List[str]): The list of attribute names used for embedding.

    Returns:
        Dict[str, float]: A dictionary where keys are attribute names and values are
                          the aggregated (averaged) scores for that attribute across all
                          reviews for the place. Returns all 0s if no reviews.
    """
    if not place_reviews:
        print("No reviews provided for this place. Returning zero attribute profile.")
        return {attr: 0.0 for attr in predefined_attributes}

    # Use defaultdict to easily sum scores for each attribute
    attribute_sums = defaultdict(float)
    review_count_with_content = 0

    for review in place_reviews:
        review_text = review.get('text', '')
        if review_text and review_text.strip(): # Only process non-empty reviews
            scores = preference_embedder_instance.get_review_attribute_scores(review_text)
            for attr, score in scores.items():
                attribute_sums[attr] += score
            review_count_with_content += 1

    place_attribute_profile = {}
    if review_count_with_content > 0:
        for attr in predefined_attributes:
            # Calculate average score for each attribute
            place_attribute_profile[attr] = attribute_sums[attr] / review_count_with_content
    else:
        # If all reviews were empty, return a profile with all zeros
        place_attribute_profile = {attr: 0.0 for attr in predefined_attributes}

    return place_attribute_profile


def calculate_preference_score(
    user_preferences: Dict[str, float],
    place_attribute_profile: Dict[str, float],
    predefined_attributes: List[str]
) -> float:
    """
    Calculates a single preference score for a place based on user's weighted
    preferences and the place's attribute profile.

    Args:
        user_preferences (Dict[str, float]): A dictionary where keys are attribute names
                                              and values are the user's desired weights/percentages
                                              for those attributes (e.g., {'Cozy': 0.7, 'Romantic': 0.3}).
                                              Weights should ideally sum to 1.0, but the function
                                              will normalize them if they don't.
        place_attribute_profile (Dict[str, float]): A dictionary where keys are attribute names
                                                    and values are the aggregated scores for that
                                                    attribute in the place's reviews (0.0 to 1.0).
        predefined_attributes (List[str]): The complete list of all possible predefined attributes.
                                           Used to ensure all attributes are considered, even if
                                           not explicitly in user_preferences or place_profile.

    Returns:
        float: The overall preference score for the place (0.0 to 1.0).
    """
    total_score = 0.0
    total_user_weight = sum(user_preferences.values())

    # Normalize user preferences if they don't sum to 1.0
    normalized_user_preferences = {}
    if total_user_weight > 0:
        for attr, weight in user_preferences.items():
            normalized_user_preferences[attr] = weight / total_user_weight
    else:
        # If no user preferences are specified, perhaps return a neutral score or 0
        # For this prototype, we'll return 0 if no preferences are given.
        print("Warning: No user preferences provided for preference score calculation. Returning 0.")
        return 0.0

    for attr in predefined_attributes:
        user_weight = normalized_user_preferences.get(attr, 0.0) # Get user's weight for this attribute, default to 0
        place_attr_score = place_attribute_profile.get(attr, 0.0) # Get place's score for this attribute, default to 0

        total_score += user_weight * place_attr_score

    return total_score

# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    # This part would typically be called from a higher-level script (e.g., cli.py or main.py)
    # and would use actual data loaded from `data/processed/`

    # 1. Initialize the PreferenceEmbedder and generate attribute embeddings
    # Define the expanded list of predefined attributes
    predefined_attributes = [
        "Cozy", "Trendy", "Romantic", "Lively", "Quiet", "Elegant", "Casual", "artistic",
        "Bohemian", "Family-Friendly", "Pet-Friendly", "Outdoor Seating", "Good for Groups",
        "Good for Solo", "Gourmet", "Comfort Food", "Healthy", "Vegan-Friendly", "Dessert",
        "Coffee", "Date", "Scenic View", "Parking Available", "Wheelchair Accessible",
        "Wi-Fi Available", "Workspace"
    ]
    embedder = PreferenceEmbedder(model_name='paraphrase-multilingual-MiniLM-L12-v2') # Changed model to a multilingual sentence embedding model
    embedder.generate_attribute_embeddings(predefined_attributes)

    # 2. Simulate reviews for a place (e.g., fetched from data_collection and processed)
    # In a real scenario, place_reviews would come from your loaded data.
    place_reviews_cafe_a = [
        {'text': 'This cafe is super cozy, with soft lighting and comfortable chairs. Perfect for a quiet afternoon.'},
        {'text': 'Great coffee, but it can get quite lively on weekends. Not really romantic, but good for friends.'},
        {'text': 'I love the cozy vibe here. It feels very welcoming and calm.'},
        {'text': 'The staff is friendly, and the atmosphere is very family-friendly, lots of space for kids.'},
        {'text': 'Good Wi-Fi and plenty of power outlets, ideal for getting some work done.'}
    ]

    place_reviews_restaurant_b = [
        {'text': 'Very trendy restaurant with a modern minimalist design. A bit noisy but the food is excellent.'},
        {'text': 'Definitely a lively spot, great for a night out with friends. Not cozy at all.'},
        {'text': 'The ambiance is chic and trendy, but not really romantic. More for a fun, energetic crowd.'},
        {'text': 'The gourmet dishes were exquisite, truly a fine dining experience.'}
    ]

    place_reviews_empty = [] # No reviews for this place
    place_reviews_only_empty_text = [{'text': ''}, {'text': '   '}, {'text': None}] # Reviews with no content

    print("\n--- Place Attribute Profiles ---")

    profile_cafe_a = calculate_place_attribute_profile(place_reviews_cafe_a, embedder, predefined_attributes)
    print(f"\nProfile for Cafe A: {profile_cafe_a}")

    profile_restaurant_b = calculate_place_attribute_profile(place_reviews_restaurant_b, embedder, predefined_attributes)
    print(f"\nProfile for Restaurant B: {profile_restaurant_b}")

    profile_empty = calculate_place_attribute_profile(place_reviews_empty, embedder, predefined_attributes)
    print(f"\nProfile for Empty Reviews Place: {profile_empty}")

    profile_only_empty_text = calculate_place_attribute_profile(place_reviews_only_empty_text, embedder, predefined_attributes)
    print(f"\nProfile for Only Empty Text Reviews Place: {profile_only_empty_text}")

    # --- New Example Usage for calculate_preference_score ---
    print("\n--- Preference Scores ---")

    # Example User 1: Prefers Cozy and Quiet
    user_prefs_1 = {"Cozy": 0.6, "Quiet": 0.4}
    score_cafe_a_user_1 = calculate_preference_score(user_prefs_1, profile_cafe_a, predefined_attributes)
    score_restaurant_b_user_1 = calculate_preference_score(user_prefs_1, profile_restaurant_b, predefined_attributes)
    print(f"\nUser 1 (Cozy/Quiet) - Score for Cafe A: {score_cafe_a_user_1:.4f}")
    print(f"User 1 (Cozy/Quiet) - Score for Restaurant B: {score_restaurant_b_user_1:.4f}")

    # Example User 2: Prefers Trendy and Lively
    user_prefs_2 = {"Trendy": 0.5, "Lively": 0.5}
    score_cafe_a_user_2 = calculate_preference_score(user_prefs_2, profile_cafe_a, predefined_attributes)
    score_restaurant_b_user_2 = calculate_preference_score(user_prefs_2, profile_restaurant_b, predefined_attributes)
    print(f"\nUser 2 (Trendy/Lively) - Score for Cafe A: {score_cafe_a_user_2:.4f}")
    print(f"User 2 (Trendy/Lively) - Score for Restaurant B: {score_restaurant_b_user_2:.4f}")

    # Example User 3: Prefers Romantic
    user_prefs_3 = {"Romantic": 1.0}
    score_cafe_a_user_3 = calculate_preference_score(user_prefs_3, profile_cafe_a, predefined_attributes)
    score_restaurant_b_user_3 = calculate_preference_score(user_prefs_3, profile_restaurant_b, predefined_attributes)
    print(f"\nUser 3 (Romantic) - Score for Cafe A: {score_cafe_a_user_3:.4f}")
    print(f"User 3 (Romantic) - Score for Restaurant B: {score_restaurant_b_user_3:.4f}")

    # Example User 4: Prefers a mix, with some attributes not present in profiles
    user_prefs_4 = {"Cozy": 0.3, "Gourmet": 0.4, "Parking Available": 0.3}
    score_cafe_a_user_4 = calculate_preference_score(user_prefs_4, profile_cafe_a, predefined_attributes)
    score_restaurant_b_user_4 = calculate_preference_score(user_prefs_4, profile_restaurant_b, predefined_attributes)
    print(f"\nUser 4 (Cozy/Gourmet/Parking) - Score for Cafe A: {score_cafe_a_user_4:.4f}")
    print(f"User 4 (Cozy/Gourmet/Parking) - Score for Restaurant B: {score_restaurant_b_user_4:.4f}")

    # Example User 5: No preferences specified (should return 0)
    user_prefs_5 = {}
    score_cafe_a_user_5 = calculate_preference_score(user_prefs_5, profile_cafe_a, predefined_attributes)
    print(f"\nUser 5 (No Preferences) - Score for Cafe A: {score_cafe_a_user_5:.4f}")
