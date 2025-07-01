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
