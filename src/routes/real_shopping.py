"""
Real shopping routes - Only web-scraped data, no mock/local data
"""

from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import json
import time
import re
from datetime import datetime
from src.services.duckpy_shopping_search import duckpy_shopping_service # Import duckpy service

real_shopping_bp = Blueprint('real_shopping', __name__)

# In-memory storage for shopping lists (in production, use a database)
shopping_lists = {}

@real_shopping_bp.route('/search', methods=['POST'])
@cross_origin()
def search_real_products():
    """Search for real products using DuckDuckGo Shopping""" # Updated docstring
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        limit = data.get('limit', 10)
        
        print(f"Searching for real products: '{query}'")
        
        # Use DuckPy Shopping service
        products = duckpy_shopping_service.search_products(query, limit) # Use duckpy service
        
        return jsonify({
            'query': query,
            'products': products,
            'total_found': len(products),
            'source': 'duckduckgo_shopping', # Updated source
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error in search_real_products: {str(e)}")
        return jsonify({'error': str(e)}), 500

@real_shopping_bp.route('/compare/<product_name>', methods=['GET'])
@cross_origin()
def compare_real_prices(product_name):
    """Compare real prices for a product across different sellers"""
    try:
        print(f"Comparing prices for: '{product_name}'")
        
        # Get price comparison from DuckPy Shopping
        comparison = duckpy_shopping_service.get_price_comparison(product_name) # Use duckpy service
        
        if 'error' in comparison:
            return jsonify(comparison), 404
        
        return jsonify(comparison)
    
    except Exception as e:
        print(f"Error in compare_real_prices: {str(e)}")
        return jsonify({'error': str(e)}), 500

@real_shopping_bp.route('/search/multiple', methods=['POST'])
@cross_origin()
def search_multiple_products():
    """Search for multiple products at once"""
    try:
        data = request.get_json()
        if not data or 'queries' not in data:
            return jsonify({'error': 'Queries list is required'}), 400
        
        queries = data['queries']
        if not isinstance(queries, list):
            return jsonify({'error': 'Queries must be a list'}), 400
        
        limit_per_query = data.get('limit', 5)
        
        results = {}
        
        for query in queries:
            if query.strip():
                print(f"Searching for: '{query}'")
                # Use DuckPy Shopping service
                products = duckpy_shopping_service.search_products(query.strip(), limit_per_query) # Use duckpy service
                results[query] = {
                    'products': products,
                    'count': len(products)
                }
        
        return jsonify({
            'results': results,
            'total_queries': len(queries),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error in search_multiple_products: {str(e)}")
        return jsonify({'error': str(e)}), 500

@real_shopping_bp.route('/list', methods=['POST'])
@cross_origin()
def create_shopping_list():
    """Create a new shopping list"""
    try:
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({'error': 'List name is required'}), 400
        
        list_name = data['name'].strip()
        if not list_name:
            return jsonify({'error': 'List name cannot be empty'}), 400
        
        # Generate unique ID
        import hashlib
        list_id = hashlib.md5(f"{list_name}_{time.time()}".encode()).hexdigest()[:8]
        
        # Create new shopping list
        shopping_lists[list_id] = {
            'id': list_id,
            'name': list_name,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'items': []
        }
        
        return jsonify({
            'message': 'Shopping list created successfully',
            'list': shopping_lists[list_id]
        })
    
    except Exception as e:
        print(f"Error in create_shopping_list: {str(e)}")
        return jsonify({'error': str(e)}), 500

@real_shopping_bp.route('/list/<list_id>/add', methods=['POST'])
@cross_origin()
def add_to_shopping_list(list_id):
    """Add real product to shopping list"""
    try:
        if list_id not in shopping_lists:
            return jsonify({'error': 'Shopping list not found'}), 404
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        quantity = data.get('quantity', 1)
        
        # Search for the product using DuckPy service
        products = duckpy_shopping_service.search_products(query, limit=1) # Use duckpy service
        
        if not products:
            return jsonify({'error': f'No products found for: {query}'}), 404
        
        best_match = products[0]

        # Validate product_url
        product_url = best_match.get('product_url', '')
        if not (isinstance(product_url, str) and product_url.startswith(('http://', 'https://'))):
            return jsonify({'error': 'No valid product link found for this item.'}), 400

        # Add to shopping list
        list_item = {
            'name': best_match['name'],
            'price': best_match['price'],
            'seller': best_match['seller'],
            'product_url': best_match['product_url'],
            'quantity': quantity,
            'category': best_match['category'],
            'added_at': datetime.now().isoformat(),
            'source_query': query
        }
        
        shopping_lists[list_id]['items'].append(list_item)
        shopping_lists[list_id]['updated_at'] = datetime.now().isoformat()
        
        return jsonify({
            'message': 'Product added to shopping list',
            'added_item': list_item,
            'list': shopping_lists[list_id]
        })
    
    except Exception as e:
        print(f"Error in add_to_shopping_list: {str(e)}")
        return jsonify({'error': str(e)}), 500

@real_shopping_bp.route('/list/<list_id>', methods=['GET'])
@cross_origin()
def get_shopping_list(list_id):
    """Get shopping list by ID"""
    try:
        if list_id not in shopping_lists:
            return jsonify({'error': 'Shopping list not found'}), 404
        
        return jsonify({'list': shopping_lists[list_id]})
    
    except Exception as e:
        print(f"Error in get_shopping_list: {str(e)}")
        return jsonify({'error': str(e)}), 500

@real_shopping_bp.route('/lists', methods=['GET'])
@cross_origin()
def get_all_shopping_lists():
    """Get all shopping lists"""
    try:
        return jsonify({'lists': list(shopping_lists.values())})
    
    except Exception as e:
        print(f"Error in get_all_shopping_lists: {str(e)}")
        return jsonify({'error': str(e)}), 500

@real_shopping_bp.route('/list/<list_id>/remove/<int:item_index>', methods=['DELETE'])
@cross_origin()
def remove_from_shopping_list(list_id, item_index):
    """Remove item from shopping list"""
    try:
        if list_id not in shopping_lists:
            return jsonify({'error': 'Shopping list not found'}), 404
        
        shopping_list = shopping_lists[list_id]
        
        if item_index < 0 or item_index >= len(shopping_list['items']):
            return jsonify({'error': 'Invalid item index'}), 400
        
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

@real_shopping_bp.route('/deals', methods=['GET'])
@cross_origin()
def get_real_deals():
    """Get real deals by searching for common grocery items"""
    try:
        # Common grocery items to search for deals
        common_items = ['milk', 'bread', 'rice', 'eggs', 'chicken', 'bananas', 'apples', 'cheese']
        
        deals = []
        
        for item in common_items[:4]:  # Limit to 4 items to avoid too many requests
            try:
                # Search DuckPy for deals
                products = duckpy_shopping_service.search_products(f"{item} deals", limit=1) # Use duckpy service
                if products:
                    deals.append(products[0])
            except Exception as e:
                print(f"Error searching for deal on {item}: {str(e)}")
        
        return jsonify({'deals': deals})
    
    except Exception as e:
        print(f"Error in get_real_deals: {str(e)}")
        return jsonify({'error': str(e)}), 500

@real_shopping_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    return {
        'status': 'healthy',
        'service': 'Real Shopping API',
        'features': ['DuckDuckGo Shopping', 'Real Products Only'] # Updated feature name
    }