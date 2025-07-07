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
