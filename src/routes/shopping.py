from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import json
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from difflib import SequenceMatcher
import requests
from langdetect import detect
import hashlib
import time
import re
from datetime import datetime
# Import services directly
from src.services.duckduckgo_shopping_search import DuckDuckGoShoppingService
from src.services.product_search import ProductSearchService

# Initialize services
duckduckgo_shopping_service = DuckDuckGoShoppingService()
product_search_service = ProductSearchService()

shopping_bp = Blueprint('shopping', __name__)

# Global variables for shopping list functionality
shopping_embedding_model = None
shopping_vector_index = None
shopping_items_db = None
shopping_embedding_to_item_map = None
shopping_lists = {}  # In-memory storage for shopping lists

def initialize_shopping_models():
    """Initialize the shopping-specific models and data."""
    global shopping_embedding_model, shopping_vector_index, shopping_items_db
    
    print("Initializing shopping models...")
    
    # Use multilingual model for better cross-language understanding
    shopping_embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # Load or create shopping items database
    load_or_create_shopping_items_db()
    
    print("Shopping models initialized successfully!")

def load_or_create_shopping_items_db():
    """Load existing shopping items database or create a new one."""
    global shopping_vector_index, shopping_items_db, shopping_embedding_to_item_map
    
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'shopping_items.pkl')
    index_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'shopping_index.faiss')
    map_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'shopping_map.pkl')
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    if os.path.exists(db_path) and os.path.exists(index_path) and os.path.exists(map_path):
        # Load existing database
        with open(db_path, 'rb') as f:
            shopping_items_db = pickle.load(f)
        shopping_vector_index = faiss.read_index(index_path)
        with open(map_path, 'rb') as f:
            shopping_embedding_to_item_map = pickle.load(f)
        print(f"Loaded {len(shopping_items_db)} shopping items from database")
    else:
        # Create new database with comprehensive shopping items
        create_shopping_items_database()

def create_shopping_items_database():
    """Create a comprehensive shopping items database with multilingual support."""
    global shopping_vector_index, shopping_items_db, shopping_embedding_model, shopping_embedding_to_item_map
    
    # Comprehensive shopping items database
    shopping_items_db = [
        # Vegetables
        {
            "id": "veg_001",
            "name": "Spinach",
            "hindi_names": ["palak", "palakh", "paalak"],
            "category": "Vegetables",
            "subcategory": "Leafy Greens",
            "common_brands": ["Fresh", "Organic", "Local"],
            "typical_units": ["bunch", "kg", "250g"],
            "keywords": ["green", "leafy", "iron", "healthy"],
            "seasonal": True,
            "price_range": "₹20-40"
        },
        {
            "id": "veg_002", 
            "name": "Okra",
            "hindi_names": ["bhindi", "bindi", "bhendi"],
            "category": "Vegetables",
            "subcategory": "Pod Vegetables",
            "common_brands": ["Fresh", "Organic"],
            "typical_units": ["500g", "1kg"],
            "keywords": ["green", "pods", "curry"],
            "seasonal": True,
            "price_range": "₹30-60"
        },
        {
            "id": "veg_003",
            "name": "Bitter Gourd",
            "hindi_names": ["karela", "karella", "karaila"],
            "category": "Vegetables", 
            "subcategory": "Gourds",
            "common_brands": ["Fresh", "Organic"],
            "typical_units": ["500g", "1kg"],
            "keywords": ["bitter", "green", "healthy", "diabetes"],
            "seasonal": True,
            "price_range": "₹40-80"
        },
        {
            "id": "veg_004",
            "name": "Fenugreek Leaves",
            "hindi_names": ["methi", "methhi", "methi patta"],
            "category": "Vegetables",
            "subcategory": "Leafy Greens",
            "common_brands": ["Fresh", "Organic"],
            "typical_units": ["bunch", "250g"],
            "keywords": ["aromatic", "herbs", "curry"],
            "seasonal": True,
            "price_range": "₹15-30"
        },
        {
            "id": "veg_005",
            "name": "Potato",
            "hindi_names": ["aloo", "alu", "aaloo"],
            "category": "Vegetables",
            "subcategory": "Root Vegetables",
            "common_brands": ["Fresh", "Local"],
            "typical_units": ["1kg", "2kg", "5kg"],
            "keywords": ["staple", "versatile", "starch"],
            "seasonal": False,
            "price_range": "₹20-40"
        },
        # Dairy Products
        {
            "id": "dairy_001",
            "name": "Milk",
            "hindi_names": ["doodh", "dudh", "paal", "pal"],
            "category": "Dairy",
            "subcategory": "Fresh Milk",
            "common_brands": ["Amul", "Mother Dairy", "Nestle", "Local"],
            "typical_units": ["500ml", "1L", "2L"],
            "keywords": ["fresh", "calcium", "protein", "daily"],
            "seasonal": False,
            "price_range": "₹25-60"
        },
        {
            "id": "dairy_002",
            "name": "Cottage Cheese",
            "hindi_names": ["paneer", "panner"],
            "category": "Dairy",
            "subcategory": "Cheese",
            "common_brands": ["Amul", "Mother Dairy", "Britannia", "Fresh"],
            "typical_units": ["200g", "500g", "1kg"],
            "keywords": ["protein", "vegetarian", "curry"],
            "seasonal": False,
            "price_range": "₹80-200"
        },
        # Grains & Cereals
        {
            "id": "grain_001",
            "name": "Rice",
            "hindi_names": ["chawal", "chaawal", "chaval"],
            "category": "Grains",
            "subcategory": "Cereals",
            "common_brands": ["India Gate", "Kohinoor", "Daawat", "Local"],
            "typical_units": ["1kg", "5kg", "10kg", "25kg"],
            "keywords": ["staple", "basmati", "long grain", "daily"],
            "seasonal": False,
            "price_range": "₹60-300"
        },
        # Herbs & Spices
        {
            "id": "herb_001",
            "name": "Coriander",
            "hindi_names": ["kothimbir", "kothimbeer", "dhaniya patta"],
            "category": "Herbs",
            "subcategory": "Fresh Herbs",
            "common_brands": ["Fresh", "Organic"],
            "typical_units": ["bunch", "100g"],
            "keywords": ["aromatic", "garnish", "fresh"],
            "seasonal": True,
            "price_range": "₹10-25"
        }
    ]
    
    # Create embeddings for all items
    all_texts = []
    shopping_embedding_to_item_map = []
    
    for item_idx, item in enumerate(shopping_items_db):
        texts_for_item = []
        
        # Main name
        texts_for_item.append(item["name"])
        
        # Hindi names
        for hindi_name in item["hindi_names"]:
            texts_for_item.append(hindi_name)
        
        # Category and subcategory
        texts_for_item.append(f"{item['category']} {item['subcategory']}")
        
        # Keywords
        texts_for_item.append(" ".join(item["keywords"]))
        
        # Combined description
        combined_desc = f"{item['name']} {' '.join(item['hindi_names'])} {item['category']} {' '.join(item['keywords'])}"
        texts_for_item.append(combined_desc)
        
        # Add all texts and map them to this item
        for text in texts_for_item:
            all_texts.append(text)
            shopping_embedding_to_item_map.append(item_idx)
    
    # Generate embeddings
    embeddings = shopping_embedding_model.encode(all_texts)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    shopping_vector_index = faiss.IndexFlatIP(dimension)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    shopping_vector_index.add(embeddings.astype('float32'))
    
    # Save database, index, and mapping
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'shopping_items.pkl')
    index_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'shopping_index.faiss')
    map_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'shopping_map.pkl')
    
    with open(db_path, 'wb') as f:
        pickle.dump(shopping_items_db, f)
    faiss.write_index(shopping_vector_index, index_path)
    with open(map_path, 'wb') as f:
        pickle.dump(shopping_embedding_to_item_map, f)
    
    print(f"Created shopping database with {len(shopping_items_db)} items")

def search_shopping_items(query, top_k=10, category_filter=None):
    """Search for shopping items using semantic similarity."""
    global shopping_embedding_model, shopping_vector_index, shopping_items_db, shopping_embedding_to_item_map
    
    if shopping_embedding_model is None or shopping_vector_index is None:
        return []
    
    # Generate query embedding
    query_embedding = shopping_embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search for similar items
    scores, indices = shopping_vector_index.search(query_embedding.astype('float32'), top_k * 3)
    
    # Map indices back to shopping items
    found_items = []
    seen_items = set()
    
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(shopping_embedding_to_item_map):
            item_idx = shopping_embedding_to_item_map[idx]
            if item_idx not in seen_items and item_idx < len(shopping_items_db):
                item = shopping_items_db[item_idx].copy()
                item['similarity_score'] = float(score)
                
                # Apply category filter if specified
                if category_filter is None or item['category'].lower() == category_filter.lower():
                    found_items.append(item)
                    seen_items.add(item_idx)
                
                if len(found_items) >= top_k:
                    break
    
    return found_items

# API Routes
@shopping_bp.route('/search', methods=['POST'])
@cross_origin()
def search_items():
    """Search for shopping items based on query."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Initialize models if not already done
        if shopping_embedding_model is None:
            initialize_shopping_models()
        
        # Get optional parameters
        top_k = data.get('limit', 10)
        category_filter = data.get('category')
        
        # Search for items
        items = search_shopping_items(query, top_k, category_filter)
        
        return jsonify({
            'query': query,
            'items': items,
            'total_found': len(items)
        })
    
    except Exception as e:
        print(f"Error in search_items: {str(e)}")
        return jsonify({'error': str(e)}), 500



@shopping_bp.route('/search/duckduckgo', methods=['POST'])
@cross_origin()
def search_duckduckgo_products():
    """Search DuckDuckGo Shopping for real products with purchase links."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        limit = data.get('limit', 10)
        
        # Use DuckDuckGo Shopping service
        products = duckduckgo_shopping_service.search_products(query, limit)
        
        return jsonify({
            'query': query,
            'products': products,
            'total_found': len(products),
            'source': 'duckduckgo_shopping'
        })
    
    except Exception as e:
        print(f"Error in search_duckduckgo_products: {str(e)}")
        return jsonify({'error': str(e)}), 500

@shopping_bp.route('/search/online', methods=['POST'])
@cross_origin()
def search_online_products():
    """Search for products across online platforms."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        limit = data.get('limit', 5)
        
        # Search across all platforms
        online_results = product_search_service.search_all_platforms(query, limit)
        
        # Get best deals comparison
        best_deals = product_search_service.get_best_deals(query)
        
        return jsonify({
            'query': query,
            'platforms': online_results,
            'best_deals': best_deals['best_deals'],
            'price_comparison': best_deals['price_comparison'],
            'total_products_found': best_deals['total_products_found']
        })
    
    except Exception as e:
        print(f"Error in search_online_products: {str(e)}")
        return jsonify({'error': str(e)}), 500

@shopping_bp.route('/list', methods=['POST'])
@cross_origin()
def create_shopping_list():
    """Create a new shopping list."""
    try:
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({'error': 'List name is required'}), 400
        
        list_name = data['name'].strip()
        if not list_name:
            return jsonify({'error': 'List name cannot be empty'}), 400
        
        # Generate unique ID
        list_id = hashlib.md5(f"{list_name}_{time.time()}".encode()).hexdigest()[:8]
        
        # Create new shopping list
        shopping_lists[list_id] = {
            'id': list_id,
            'name': list_name,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'items': [],
            'total_estimated_cost': 0
        }
        
        return jsonify({
            'message': 'Shopping list created successfully',
            'list': shopping_lists[list_id]
        })
    
    except Exception as e:
        print(f"Error in create_shopping_list: {str(e)}")
        return jsonify({'error': str(e)}), 500

@shopping_bp.route('/list/<list_id>/add', methods=['POST'])
@cross_origin()
def add_to_shopping_list(list_id):
    """Add items to shopping list."""
    try:
        if list_id not in shopping_lists:
            return jsonify({'error': 'Shopping list not found'}), 404
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        quantity = data.get('quantity', 1)
        
        # Initialize models if not already done
        if shopping_embedding_model is None:
            initialize_shopping_models()
        
        # Search for the item
        items = search_shopping_items(query, top_k=1)
        
        if not items:
            return jsonify({'error': f'No items found for query: {query}'}), 404
        
        best_match = items[0]
        
        # Add to shopping list
        list_item = {
            'item_id': best_match['id'],
            'name': best_match['name'],
            'hindi_names': best_match['hindi_names'],
            'category': best_match['category'],
            'quantity': quantity,
            'unit': best_match['typical_units'][0] if best_match['typical_units'] else 'piece',
            'estimated_price': best_match['price_range'],
            'added_at': datetime.now().isoformat(),
            'similarity_score': best_match['similarity_score']
        }
        
        shopping_lists[list_id]['items'].append(list_item)
        shopping_lists[list_id]['updated_at'] = datetime.now().isoformat()
        
        return jsonify({
            'message': 'Item added to shopping list',
            'added_item': list_item,
            'list': shopping_lists[list_id]
        })
    
    except Exception as e:
        print(f"Error in add_to_shopping_list: {str(e)}")
        return jsonify({'error': str(e)}), 500

@shopping_bp.route('/search/combined-real', methods=['POST'])
@cross_origin()
def search_combined_real():
    """Search both local database and real online products via DuckDuckGo."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Initialize models if not already done
        if shopping_embedding_model is None:
            initialize_shopping_models()
        
        limit = data.get('limit', 5)
        
        # Search local database
        local_items = search_shopping_items(query, top_k=limit)
        
        # Search DuckDuckGo Shopping for real products
        duckduckgo_products = duckduckgo_shopping_service.search_products(query, limit)
        
        # Get price comparison
        price_comparison = duckduckgo_shopping_service.get_price_comparison(query)
        
        return jsonify({
            'query': query,
            'local_items': local_items,
            'duckduckgo_products': duckduckgo_products,
            'price_comparison': price_comparison,
            'recommendations': {
                'best_local': local_items[0] if local_items else None,
                'best_online': duckduckgo_products[0] if duckduckgo_products else None,
                'cheapest_online': price_comparison.get('recommendations', {}).get('cheapest') if 'error' not in price_comparison else None
            }
        })
    
    except Exception as e:
        print(f"Error in add_to_shopping_list: {str(e)}")
        return jsonify({'error': str(e)}), 500

@shopping_bp.route('/list/<list_id>', methods=['GET'])
@cross_origin()
def get_shopping_list(list_id):
    """Get shopping list by ID."""
    try:
        if list_id not in shopping_lists:
            return jsonify({'error': 'Shopping list not found'}), 404
        
        return jsonify({'list': shopping_lists[list_id]})
    
    except Exception as e:
        print(f"Error in get_shopping_list: {str(e)}")
        return jsonify({'error': str(e)}), 500

@shopping_bp.route('/list/<list_id>/remove/<int:item_index>', methods=['DELETE'])
@cross_origin()
def remove_from_shopping_list(list_id, item_index):
    """Remove item from shopping list by index."""
    try:
        if list_id not in shopping_lists:
            return jsonify({'error': 'Shopping list not found'}), 404
        
        shopping_list = shopping_lists[list_id]
        
        if item_index < 0 or item_index >= len(shopping_list['items']):
            return jsonify({'error': 'Invalid item index'}), 400
        
        # Remove the item
        removed_item = shopping_list['items'].pop(item_index)
        shopping_list['updated_at'] = datetime.now().isoformat()
        
        return jsonify({
            'message': 'Item removed from shopping list',
            'removed_item': removed_item,
            'list': shopping_list
        })
    
    except Exception as e:
        print(f"Error in remove_from_shopping_list: {str(e)}")
        return jsonify({'error': str(e)}), 500

@shopping_bp.route('/list/<list_id>/clear', methods=['DELETE'])
@cross_origin()
def clear_shopping_list(list_id):
    """Clear all items from shopping list."""
    try:
        if list_id not in shopping_lists:
            return jsonify({'error': 'Shopping list not found'}), 404
        
        shopping_list = shopping_lists[list_id]
        shopping_list['items'] = []
        shopping_list['updated_at'] = datetime.now().isoformat()
        shopping_list['total_estimated_cost'] = 0
        
        return jsonify({
            'message': 'Shopping list cleared',
            'list': shopping_list
        })
    
    except Exception as e:
        print(f"Error in clear_shopping_list: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Initialize models when the module is imported
try:
    initialize_shopping_models()
except Exception as e:
    print(f"Warning: Could not initialize shopping models: {e}")