# AI-Powered Place Recommendation System

## Overview

This project is an AI-powered recommendation system designed to suggest restaurants, cafes, or other places to users based on a comprehensive set of criteria. The system aims to provide personalized recommendations by considering user preferences, the sentiment of reviews, geographical proximity, star ratings, and budget compatibility.

This repository contains the prototype implementation, focusing on demonstrating the core recommendation logic and multi-source data integration.

## Project Structure 
```bash
ai-place-recommender/
├── src/
│   ├── data_collection/
│   │   ├── google_maps_collector.py  # Collects place & review data from Google Maps
│   │   ├── external_source_collector.py # Collects data from Yelp/TripAdvisor/etc.
│   │   └── __init__.py
│   ├── data_processing/
│   │   ├── data_normalizer.py      # Cleans and unifies data from various sources
│   │   └── __init__.py
│   ├── nlp_models/
│   │   ├── sentiment_analyzer.py   # Performs sentiment analysis on reviews
│   │   ├── preference_embedder.py  # Generates embeddings for preferences/reviews
│   │   └── __init__.py
│   ├── core_logic/
│   │   ├── scoring.py            # Calculates individual scores (distance, stars, budget, etc.)
│   │   ├── ranking.py            # Applies weighted equation and ranks places
│   │   └── __init__.py
│   └── ui/
│       ├── cli.py                # Command-Line Interface for user interaction
│       └── __init__.py
├── data/
│   ├── raw/                      # Raw collected data (e.g., from APIs before normalization)
│   ├── processed/                # Normalized, ready-to-use data (for prototype, could be JSON/CSV)
│   └── placeholder_data/         # Initial synthetic data for testing before real collection
│       ├── places.json
│       └── reviews.json
├── config/
│   └── settings.py               # Stores configuration like model paths, initial weights, etc.
├── tests/
│   ├── test_data_processing.py
│   ├── test_nlp_models.py
│   ├── test_core_logic.py
│   └── __init__.py
├── .env.example                  # Example for environment variables (API keys)
├── .gitignore                    # Specifies intentionally untracked files to ignore
├── requirements.txt              # Lists Python dependencies
└── README.md                     # Project overview and instructions
```

## Features

* **Multi-Source Data Integration:** Gathers place and review data from Google Maps Platform and at least one additional external source (e.g., Yelp).
* **User Preference Input:** Allows users to specify their desired place attributes (e.g., "Cozy", "Trendy", "Romantic") with customizable percentage weights.
* **Sentiment Analysis:** Analyzes the sentiment (positive/negative) of user reviews for each place.
* **Preference Similarity:** Calculates how well a place's attributes (derived from reviews) match the user's weighted preferences using contextual embeddings.
* **Location-Based Scoring:** Determines a score based on the distance between the user and the place (utilizing Google Maps Distance Matrix API for real travel distances).
* **Star Rating Integration:** Incorporates the average star rating of a place into the overall score.
* **Budget Matching:** Assesses how well a place's typical budget aligns with the user's specified budget range.
* **Weighted Ranking Equation:** Combines all individual scores into a single, customizable weighted formula to generate a ranked list of recommendations.

## Architecture (High-Level)

The system is structured into modular components:

1.  **Data Collection:** Scripts to fetch raw place and review data from various online platforms (Google Maps, Yelp, etc.).
2.  **Data Processing:** Modules for cleaning, normalizing, and unifying the collected data into a consistent format.
3.  **NLP Models:** Components for performing sentiment analysis on review text and generating contextual embeddings for preference matching.
4.  **Core Logic:** Functions responsible for calculating individual scores (distance, stars, budget, sentiment, preference similarity) and then combining them using a weighted ranking algorithm.
5.  **User Interface (UI):** A command-line interface for users to input their preferences and view recommendations.

## Getting Started

Follow these steps to set up and run the prototype locally.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### 1. Clone the Repository

```bash
git clone https://github.com/haneenalaa465/AI-Powered-Place-Recommendation-System
cd ai-place-recommender
```

### 2. Install Dependencies
It's highly recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
pip install -r requirements.txt
```
