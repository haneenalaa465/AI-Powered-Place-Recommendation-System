import json
from collections import defaultdict
from typing import List, Dict, Any

# Import PreferenceEmbedder from its module
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

# Example Usage (for testing/demonstration of scoring functions)
if __name__ == "__main__":
    # Ensure the project root is in sys.path for imports to work in Colab
    import sys
    import os
    project_root = '/content/AI-Powered-Place-Recommendation-System'
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 1. Initialize the PreferenceEmbedder and generate attribute embeddings
    predefined_attributes = [
        "Cozy", "Trendy", "Romantic", "Lively", "Quiet", "Elegant", "Casual", "artistic",
        "Bohemian", "Family-Friendly", "Pet-Friendly", "Outdoor Seating", "Good for Groups",
        "Good for Solo", "Gourmet", "Comfort Food", "Healthy", "Vegan-Friendly", "Dessert",
        "Coffee", "Date", "Scenic View", "Parking Available", "Wheelchair Accessible",
        "Wi-Fi Available", "Workspace"
    ]
    # Ensure the embedder uses the Jina model as requested
    embedder = PreferenceEmbedder(model_name='jinaai/jina-embeddings-v3')
    embedder.generate_attribute_embeddings(predefined_attributes)

    # --- Combined Review Data (English, Arabic, Tunisian, French) ---
    all_reviews_data = {
        "texts": [
            "This cafe has such a warm and inviting atmosphere, perfect for reading.",
            "المكان رايق والقعدة بتاعته حلوة، أكيد هروح تاني.",
            "Idéal pour travailler, le Wi-Fi est rapide et les sièges confortables.",
            "Super modern decor and great music, definitely a trendy spot for young people.",
            "الأكل كان تحفة والخدمة ممتازة، مكان يستاهل التجربة.",
            "Ce restaurant est très élégant et les plats sont gastronomiques.",
            "The dim lighting and soft jazz made it ideal for a romantic dinner.",
            "بلاصة ياسر مزيانة والقهوة بنينة برشا.",
            "Très bruyant et bondé, mais l'énergie était incroyable ! Idéal pour les groupes.",
            "This place is both trendy and cozy, a rare combination!",
            "الموظفين كانوا بشوشين والخدمة طيارة.",
            "L'ambiance est décontractée, parfaite pour une soirée entre amis.",
            "Excellent fine dining experience with an elegant ambiance and gourmet dishes.",
            "أنصح فيه للعائلات، فيه مساحة كبيرة للأطفال.",
            "La nourriture était saine et délicieuse, avec de nombreuses options végétaliennes.",
            "A great spot for remote work, good coffee and reliable Wi-Fi.",
            "منظر يفتح النفس والخدمة ممتازة.",
            "C'est un lieu artistique avec une décoration unique.",
            "The staff is friendly, and the atmosphere is very family-friendly, lots of space for kids.",
            "الأسعار كانت معقولة جداً والجودة عالية.",
            "Vue imprenable et service impeccable, parfait pour une date.",
            "It's a bit loud, but the food was fantastic. Not very cozy though.",
            "المكان هادئ ومريح للأعصاب، مثالي للاسترخاء.",
            "Le parking est facile d'accès, ce qui est un un plus.",
            "The coffee here is top-notch.",
            "اللقمات طيبين والمحل بجنن، أجواء رائعة.",
            "Un endroit calme pour lire un livre avec un bon café.",
            "The portions were generous and delicious.",
            "المكان جميل جداً والمنظر ساحر.",
            "Ce café accepte les animaux de compagnie, c'est génial !",
            "Perfect in every way, a truly memorable meal.",
            "الخدمة عشرة على عشرة، يستاهل كل ريال تدفعه.",
            "Les desserts sont incroyables, surtout le tiramisu.",
            "A must-visit spot, incredible value for money.",
            "أفضل مكان جربته في حياتي، كل شيء كان على أكمل وجه.",
            "Un endroit très animé, idéal pour faire la fête.",
            "The ambiance is perfect for a date night.",
            "الطاقم محترف جداً والتقديم كان احترافي.",
            "L'accès en fauteuil roulant est très bien pensé.",
            "المكان نظيف ومرتب، مستوى النظافة عالي جداً.",
            "The menu has a great variety of options.",
            "بنة على بنة، يعطيهم الصحة.",
            "المحل واعر والخدمة نقية.",
            "الجو العام للمكان لطيف.",
            "أجواء المطعم رائعة.",
            "The best view in town.",
            "مكان تشعر فيه بالترحاب.",
            "تجربة فريدة من نوعها.",
            "كل طبق كان ألذ من الثاني.",
            "I'm already planning my next visit.",
            "أحسن اختيار لمناسبة خاصة.",
            "الموقع استراتيجي ومميز.",
            "فريق عمل سريع الاستجابة.",
            "الأجواء كانت مريحة.",
            "تجربة تستحق خمس نجوم.",
            "الجودة مقابل السعر ممتازة.",
            "أكيد سأعود مرة أخرى.",
            "Les portions sont généreuses et le rapport qualité-prix est excellent.",
            "C'est un endroit parfait pour les groupes, avec beaucoup d'espace.",
            "Ce lieu est très artistique et inspirant.",
            "Le service est rapide et le personnel est très accueillant.",
            "J'ai adoré l'ambiance bohème de ce café.",
            "Les options sans gluten sont excellentes ici.",
            "Le restaurant est idéal pour les repas en solo.",
            "La terrasse extérieure est très agréable.",
            "Le café est spécialisé dans les boissons artisanales."
        ]
    }

    # Convert raw text list to the expected format [{'text': 'review content'}]
    all_reviews = [{'text': r} for r in all_reviews_data['texts']]

    # --- Simulate various places with mixed language reviews to show preferences ---

    # Place 1: Cozy & Quiet Cafe (mix of English, Arabic, French)
    place_reviews_cozy_cafe = [
        all_reviews[0],   # English: "This cafe has such a warm and inviting atmosphere, perfect for reading."
        all_reviews[1],   # Arabic: "المكان رايق والقاعدة بتاعته حلوة، أكيد هروح تاني." (Calm, nice seating -> Quiet, Cozy)
        all_reviews[22],  # Arabic: "المكان هادئ ومريح للأعصاب، مثالي للاسترخاء." (Quiet and relaxing -> Quiet, Cozy)
        all_reviews[26],  # French: "Un endroit calme pour lire un livre avec un bon café." (Quiet, Coffee, Cozy)
        all_reviews[43]   # Arabic: "الجو العام للمكان لطيف." (Nice atmosphere -> Cozy, Casual)
    ]

    # Place 2: Trendy & Lively Restaurant (mix of English, Arabic, French)
    place_reviews_trendy_restaurant = [
        all_reviews[3],   # English: "Super modern decor and great music, definitely a trendy spot for young people."
        all_reviews[8],   # French: "Très bruyant et bondé, mais l'énergie était incroyable ! Idéal pour les groupes." (Lively, Good for Groups)
        all_reviews[9],   # English: "This place is both trendy and cozy, a rare combination!"
        all_reviews[35],  # French: "Un endroit très animé, idéal pour faire la fête." (Lively)
        all_reviews[44]   # Arabic: "أجواء المطعم رائعة." (Wonderful ambiance -> Lively, Trendy)
    ]

    # Place 3: Family-Friendly & Accessible Spot (mix of English, Arabic, French)
    place_reviews_family_spot = [
        all_reviews[13],  # Arabic: "أنصح فيه للعائلات، فيه مساحة كبيرة للأطفال." (Recommend for families, space for kids -> Family-Friendly)
        all_reviews[18],  # English: "The staff is friendly, and the atmosphere is very family-friendly, lots of space for kids."
        all_reviews[59],  # French: "C'est un endroit parfait pour les groupes, avec beaucoup d'espace." (Good for Groups)
        all_reviews[63],  # French: "Le restaurant est idéal pour les repas en solo." (Good for Solo)
        all_reviews[38]   # French: "L'accès en fauteuil roulant est très bien pensé." (Wheelchair Accessible)
    ]

    # Place 4: Workspace & Coffee Spot (mix of English, Arabic, Tunisian, French)
    place_reviews_workspace_cafe = [
        all_reviews[2],   # French: "Idéal pour travailler, le Wi-Fi est rapide et les sièges confortables." (Workspace, Wi-Fi Available)
        all_reviews[15],  # English: "A great spot for remote work, good coffee and reliable Wi-Fi."
        all_reviews[24],  # English: "The coffee here is top-notch."
        all_reviews[7],   # Tunisian: "بلاصة ياسر مزيانة والقهوة بنينة برشا." (Very nice place, tasty coffee -> Coffee, Casual)
        all_reviews[41],  # Tunisian: "بنة على بنة، يعطيهم الصحة." (Delicious)
        all_reviews[65]   # French: "Le café est spécialisé dans les boissons artisanales." (Coffee)
    ]

    # Place 5: Elegant & Gourmet Dining (mix of English, Arabic, French)
    place_reviews_elegant_gourmet = [
        all_reviews[5],   # French: "Ce restaurant est très élégant et les plats sont gastronomiques." (Elegant, Gourmet)
        all_reviews[12],  # English: "Excellent fine dining experience with an elegant ambiance and gourmet dishes."
        all_reviews[27],  # English: "The portions were generous and delicious." (Gourmet, Comfort Food)
        all_reviews[37],  # Arabic: "الطاقم محترف جداً والتقديم كان احترافي." (Professional staff, professional presentation -> Elegant)
        all_reviews[48]   # Arabic: "كل طبق كان ألذ من الثاني." (Every dish was tastier than the last -> Gourmet)
    ]

    # Place 6: Romantic & Scenic View Spot
    place_reviews_romantic_scenic = [
        all_reviews[6],   # English: "The dim lighting and soft jazz made it ideal for a romantic dinner."
        all_reviews[20],  # English: "Vue imprenable et service impeccable, parfait pour une date."
        all_reviews[28],  # Arabic: "المكان جميل جداً والمنظر ساحر."
        all_reviews[36],  # English: "The ambiance is perfect for a date night."
        all_reviews[45]   # English: "The best view in town."
    ]

    # Place 7: Artistic & Bohemian Vibe
    place_reviews_artistic_bohemian = [
        all_reviews[17],  # French: "C'est un lieu artistique avec une décoration unique."
        all_reviews[60],  # French: "Ce lieu est très artistique et inspirant."
        all_reviews[62]   # French: "J'ai adoré l'ambiance bohème de ce café."
    ]


    # Place with no reviews
    place_reviews_empty = []

    # Place with reviews that are empty strings or None
    place_reviews_only_empty_text = [{'text': ''}, {'text': '   '}, {'text': None}]


    print("\n--- Place Attribute Profiles (Multi-Language Reviews) ---")

    profile_cozy_cafe = calculate_place_attribute_profile(place_reviews_cozy_cafe, embedder, predefined_attributes)
    print(f"\nProfile for Cozy & Quiet Cafe: {profile_cozy_cafe}")

    profile_trendy_restaurant = calculate_place_attribute_profile(place_reviews_trendy_restaurant, embedder, predefined_attributes)
    print(f"\nProfile for Trendy & Lively Restaurant: {profile_trendy_restaurant}")

    profile_family_spot = calculate_place_attribute_profile(place_reviews_family_spot, embedder, predefined_attributes)
    print(f"\nProfile for Family-Friendly & Accessible Spot: {profile_family_spot}")

    profile_workspace_cafe = calculate_place_attribute_profile(place_reviews_workspace_cafe, embedder, predefined_attributes)
    print(f"\nProfile for Workspace & Coffee Spot: {profile_workspace_cafe}")

    profile_elegant_gourmet = calculate_place_attribute_profile(place_reviews_elegant_gourmet, embedder, predefined_attributes)
    print(f"\nProfile for Elegant & Gourmet Dining: {profile_elegant_gourmet}")

    profile_romantic_scenic = calculate_place_attribute_profile(place_reviews_romantic_scenic, embedder, predefined_attributes)
    print(f"\nProfile for Romantic & Scenic View Spot: {profile_romantic_scenic}")

    profile_artistic_bohemian = calculate_place_attribute_profile(place_reviews_artistic_bohemian, embedder, predefined_attributes)
    print(f"\nProfile for Artistic & Bohemian Place: {profile_artistic_bohemian}")

    profile_empty = calculate_place_attribute_profile(place_reviews_empty, embedder, predefined_attributes)
    print(f"\nProfile for Empty Reviews Place: {profile_empty}")

    profile_only_empty_text = calculate_place_attribute_profile(place_reviews_only_empty_text, embedder, predefined_attributes)
    print(f"\nProfile for Only Empty Text Reviews Place: {profile_only_empty_text}")

    # --- Preference Scores with Multi-Language Profiles ---
    print("\n--- Preference Scores (Multi-Language Examples) ---")

    # User 1: Prefers Cozy and Quiet
    user_prefs_1 = {"Cozy": 0.6, "Quiet": 0.4}
    score_cozy_cafe_user_1 = calculate_preference_score(user_prefs_1, profile_cozy_cafe, predefined_attributes)
    score_trendy_restaurant_user_1 = calculate_preference_score(user_prefs_1, profile_trendy_restaurant, predefined_attributes)
    print(f"\nUser 1 (Cozy/Quiet) - Score for Cozy & Quiet Cafe: {score_cozy_cafe_user_1:.4f}")
    print(f"User 1 (Cozy/Quiet) - Score for Trendy & Lively Restaurant: {score_trendy_restaurant_user_1:.4f}")

    # User 2: Prefers Trendy and Lively
    user_prefs_2 = {"Trendy": 0.5, "Lively": 0.5}
    score_cozy_cafe_user_2 = calculate_preference_score(user_prefs_2, profile_cozy_cafe, predefined_attributes)
    score_trendy_restaurant_user_2 = calculate_preference_score(user_prefs_2, profile_trendy_restaurant, predefined_attributes)
    print(f"\nUser 2 (Trendy/Lively) - Score for Cozy & Quiet Cafe: {score_cozy_cafe_user_2:.4f}")
    print(f"User 2 (Trendy/Lively) - Score for Trendy & Lively Restaurant: {score_trendy_restaurant_user_2:.4f}")

    # User 3: Prefers Family-Friendly and Accessible
    user_prefs_3 = {"Family-Friendly": 0.7, "Wheelchair Accessible": 0.3}
    score_cozy_cafe_user_3 = calculate_preference_score(user_prefs_3, profile_cozy_cafe, predefined_attributes)
    score_family_spot_user_3 = calculate_preference_score(user_prefs_3, profile_family_spot, predefined_attributes)
    print(f"\nUser 3 (Family-Friendly/Accessible) - Score for Cozy & Quiet Cafe: {score_cozy_cafe_user_3:.4f}")
    print(f"User 3 (Family-Friendly/Accessible) - Score for Family-Friendly & Accessible Spot: {score_family_spot_user_3:.4f}")

    # User 4: Prefers Workspace and Wi-Fi
    user_prefs_4 = {"Workspace": 0.7, "Wi-Fi Available": 0.3}
    score_cozy_cafe_user_4 = calculate_preference_score(user_prefs_4, profile_cozy_cafe, predefined_attributes)
    score_workspace_cafe_user_4 = calculate_preference_score(user_prefs_4, profile_workspace_cafe, predefined_attributes)
    print(f"\nUser 4 (Workspace/Wi-Fi) - Score for Cozy & Quiet Cafe: {score_cozy_cafe_user_4:.4f}")
    print(f"User 4 (Workspace/Wi-Fi) - Score for Workspace & Coffee Spot: {score_workspace_cafe_user_4:.4f}")

    # User 5: Prefers Gourmet and Elegant
    user_prefs_5 = {"Gourmet": 0.5, "Elegant": 0.5}
    score_trendy_restaurant_user_5 = calculate_preference_score(user_prefs_5, profile_trendy_restaurant, predefined_attributes)
    score_elegant_gourmet_user_5 = calculate_preference_score(user_prefs_5, profile_elegant_gourmet, predefined_attributes)
    print(f"\nUser 5 (Gourmet/Elegant) - Score for Trendy & Lively Restaurant: {score_trendy_restaurant_user_5:.4f}")
    print(f"User 5 (Gourmet/Elegant) - Score for Elegant & Gourmet Dining: {score_elegant_gourmet_user_5:.4f}")

    # User 6: Prefers Romantic and Scenic View
    user_prefs_6 = {"Romantic": 0.6, "Scenic View": 0.4}
    score_romantic_scenic_user_6 = calculate_preference_score(user_prefs_6, profile_romantic_scenic, predefined_attributes)
    print(f"\nUser 6 (Romantic/Scenic) - Score for Romantic & Scenic View Spot: {score_romantic_scenic_user_6:.4f}")

    # User 7: Prefers Artistic and Bohemian
    user_prefs_7 = {"artistic": 0.6, "Bohemian": 0.4}
    score_artistic_bohemian_user_7 = calculate_preference_score(user_prefs_7, profile_artistic_bohemian, predefined_attributes)
    print(f"\nUser 7 (Artistic/Bohemian) - Score for Artistic & Bohemian Place: {score_artistic_bohemian_user_7:.4f}")

    # Example User 8: No preferences specified (should return 0)
    user_prefs_8 = {}
    score_cozy_cafe_user_8 = calculate_preference_score(user_prefs_8, profile_cozy_cafe, predefined_attributes)
    print(f"\nUser 8 (No Preferences) - Score for Cozy & Quiet Cafe: {score_cozy_cafe_user_8:.4f}")
