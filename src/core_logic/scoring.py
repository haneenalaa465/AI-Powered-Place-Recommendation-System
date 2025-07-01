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
