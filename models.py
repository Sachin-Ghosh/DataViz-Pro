from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Create a single MongoDB instance
mongo = PyMongo()

def init_db(app):
    global mongo
    mongo.init_app(app)

class User:
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password_hash = generate_password_hash(password)
        self.created_at = datetime.utcnow()

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @staticmethod
    def create_user(username, email, password):
        user = {
            'username': username,
            'email': email,
            'password_hash': generate_password_hash(password),
            'created_at': datetime.utcnow()
        }
        return mongo.db.users.insert_one(user)

    @staticmethod
    def get_user_by_username(username):
        return mongo.db.users.find_one({'username': username})

    @staticmethod
    def get_user_by_email(email):
        return mongo.db.users.find_one({'email': email})