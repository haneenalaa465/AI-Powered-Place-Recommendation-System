# src/core_logic/ranking.py

import math
from typing import List, Dict, Any, Tuple

# Assuming your updated SentimentAnalyzer provides scores from 0 to 1
from src.nlp_models.sentiment_analyzer import SentimentAnalyzer
from src.nlp_models.preference_embedder import PreferenceEmbedder
from src.core_logic.scoring import calculate_place_attribute_profile, calculate_preference_score

# --- Score Normalization and Calculation Helpers ---

def _calculate_proximity_score(
    user_coords: Tuple[float, float], 
    place_coords: Tuple[float, float],
    max_distance_km: int = 10
) -> float:
    """
    Calculates a normalized proximity score (0-1) based on geographic distance.
    Uses the Haversine formula. A higher score means closer.

    Args:
        user_coords (Tuple[float, float]): The user's (latitude, longitude).
        place_coords (Tuple[float, float]): The place's (latitude, longitude).
        max_distance_km (int): The distance at which the score becomes 0.

    Returns:
        float: A normalized proximity score between 0.0 and 1.0.
    """
    R = 6371  # Earth radius in kilometers
    lat1, lon1 = math.radians(user_coords[0]), math.radians(user_coords[1])
    lat2, lon2 = math.radians(place_coords[0]), math.radians(place_coords[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    if distance > max_distance_km:
        return 0.0
    
    # Normalize score: 1 for 0 distance, 0 for max_distance
    return 1.0 - (distance / max_distance_km)


def _calculate_budget_score(user_budget: int, place_budget: int) -> float:
    """
    Calculates a normalized budget match score (0-1).
    A higher score means budgets are more similar.

    Args:
        user_budget (int): User's budget level (0-3).
        place_budget (int): Place's budget level (0-3).

    Returns:
        float: A normalized budget score between 0.0 and 1.0.
    """
    # The max possible difference on a 0-3 scale is 3.
    max_budget_diff = 3.0
    diff = abs(user_budget - place_budget)
    
    # Normalize score: 1 for 0 difference, 0 for max difference
    return 1.0 - (diff / max_budget_diff)


def _calculate_aggregated_sentiment(reviews_with_sentiment: List[Dict[str, Any]]) -> float:
    """
    Calculates the average sentiment score from a list of analyzed reviews.

    Args:
        reviews_with_sentiment (List[Dict[str, Any]]): Reviews after processing
            by SentimentAnalyzer, each containing a 'sentiment_score'.

    Returns:
        float: The average sentiment score (0-1). Returns 0.5 if no reviews.
    """
    if not reviews_with_sentiment:
        return 0.5  # Return a neutral score if there are no reviews

    total_score = sum(review.get('sentiment_score', 0.5) for review in reviews_with_sentiment)
    return total_score / len(reviews_with_sentiment)


# --- Main Ranking Function ---

def calculate_final_score(
    place_data: Dict[str, Any],
    user_data: Dict[str, Any],
    embedder: PreferenceEmbedder,
    sentiment_analyzer: SentimentAnalyzer,
    predefined_attributes: List[str]
) -> Dict[str, float]:
    """
    Calculates the final weighted score for a place based on multiple factors.

    Args:
        place_data (Dict[str, Any]): Dictionary with place info, including:
            'reviews', 'budget', and 'coords'.
        user_data (Dict[str, Any]): Dictionary with user info, including:
            'preferences', 'budget', and 'coords'.
        embedder (PreferenceEmbedder): An initialized PreferenceEmbedder instance.
        sentiment_analyzer (SentimentAnalyzer): An initialized SentimentAnalyzer instance.
        predefined_attributes (List[str]): List of preference attributes.

    Returns:
        Dict[str, float]: A dictionary containing the final score and its components.
    """
    # 1. Calculate Sentiment Score (Weight: 0.25)
    reviews_with_sentiment = sentiment_analyzer.analyze_reviews(place_data['reviews'])
    sentiment_score = _calculate_aggregated_sentiment(reviews_with_sentiment)

    # 2. Calculate Preference Matching Score (Weight: 0.30)
    place_profile = calculate_place_attribute_profile(
        place_data['reviews'], embedder, predefined_attributes
    )
    preference_score = calculate_preference_score(
        user_data['preferences'], place_profile, predefined_attributes
    )
    
    # 3. Calculate Proximity Score (Weight: 0.30)
    proximity_score = _calculate_proximity_score(
        user_data['coords'], place_data['coords']
    )

    # 4. Calculate Budget Matching Score (Weight: 0.15)
    budget_score = _calculate_budget_score(
        user_data['budget'], place_data['budget']
    )

    # Apply the final weighted formula
    final_score = (
        (sentiment_score * 0.25) +
        (preference_score * 0.30) +
        (proximity_score * 0.30) +
        (budget_score * 0.15)
    )

    return {
        "final_score": round(final_score, 4),
        "sentiment_score": round(sentiment_score, 4),
        "preference_score": round(preference_score, 4),
        "proximity_score": round(proximity_score, 4),
        "budget_score": round(budget_score, 4)
    }
    
def rank_places(
    user_data: Dict[str, Any],
    places_list: List[Dict[str, Any]],
    embedder: PreferenceEmbedder,
    sentiment_analyzer: SentimentAnalyzer,
    predefined_attributes: List[str]
) -> List[Dict[str, Any]]:
    """
    Calculates scores for multiple places and returns them sorted by final score.

    Args:
        user_data (Dict[str, Any]): A dictionary containing the user's details
            (preferences, budget, coords).
        places_list (List[Dict[str, Any]]): A list of dictionaries, where each
            dictionary represents a place to be ranked.
        embedder (PreferenceEmbedder): An initialized PreferenceEmbedder instance.
        sentiment_analyzer (SentimentAnalyzer): An initialized SentimentAnalyzer instance.
        predefined_attributes (List[str]): The list of preference attributes.

    Returns:
        List[Dict[str, Any]]: The list of places, sorted in descending order
                              by their final score, with scoring details added.
    """
    scored_places = []
    print(f"\nScoring {len(places_list)} places...")

    for place in places_list:
        # Calculate the score for the current place using the existing function
        score_details = calculate_final_score(
            place_data=place,
            user_data=user_data,
            embedder=embedder,
            sentiment_analyzer=sentiment_analyzer,
            predefined_attributes=predefined_attributes
        )
        
        # Add the detailed scores to the place's data
        place['scoring_details'] = score_details
        scored_places.append(place)

    # Sort the list of places based on the nested 'final_score' in descending order
    sorted_places = sorted(
        scored_places,
        key=lambda p: p['scoring_details']['final_score'],
        reverse=True
    )
    
    print("Ranking complete.")
    return sorted_places