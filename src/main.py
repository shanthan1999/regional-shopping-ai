import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
import logging
from src.models.user import db
from src.routes.user import user_bp
from src.routes.rag import rag_bp
from src.routes.shopping import shopping_bp
from src.services.duckduckgo_shopping_search import DuckDuckGoShoppingService
from src.services.product_search import ProductSearchService
from src.config import get_config

# Get configuration
config_class = get_config()

# Configure logging
logging.basicConfig(
    level=getattr(logging, config_class.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config.from_object(config_class)

# Enable CORS for all routes
CORS(app)

# Global error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f'Server Error: {error}')
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f'Unhandled Exception: {e}')
    return jsonify({'error': 'An unexpected error occurred'}), 500

# Initialize services
duckduckgo_shopping_service = DuckDuckGoShoppingService()
product_search_service = ProductSearchService()

app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(rag_bp, url_prefix='/api/rag')
app.register_blueprint(shopping_bp, url_prefix='/api/shopping')

# uncomment if you need to use database
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'app.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
with app.app_context():
    db.create_all()

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Regional Shopping AI is running',
        'version': '1.0.0'
    })

@app.route('/api/status')
def api_status():
    """API status endpoint"""
    return jsonify({
        'api_status': 'active',
        'services': {
            'shopping': 'active',
            'rag': 'active',
            'user_management': 'active'
        },
        'endpoints': {
            'shopping_search': '/api/shopping/search',
            'duckduckgo_search': '/api/shopping/search/duckduckgo',
            'shopping_list': '/api/shopping/list',
            'rag_query': '/api/rag/ask'
        }
    })

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
