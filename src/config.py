"""
Configuration settings for Regional Shopping AI
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # API Configuration
    API_RATE_LIMIT = os.getenv('API_RATE_LIMIT', '100 per hour')
    
    # Search Configuration
    DEFAULT_SEARCH_LIMIT = int(os.getenv('DEFAULT_SEARCH_LIMIT', '10'))
    MAX_SEARCH_LIMIT = int(os.getenv('MAX_SEARCH_LIMIT', '50'))
    
    # Cache Configuration
    CACHE_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', '300'))  # 5 minutes
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Service Configuration
    ENABLE_DUCKDUCKGO_SEARCH = os.getenv('ENABLE_DUCKDUCKGO_SEARCH', 'true').lower() == 'true'
    ENABLE_RAG_SERVICE = os.getenv('ENABLE_RAG_SERVICE', 'true').lower() == 'true'
    
    # Model Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv('DEV_DATABASE_URL', 'sqlite:///dev_app.db')

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///app.db')

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])