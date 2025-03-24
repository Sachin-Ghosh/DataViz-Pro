from flask import Flask, render_template, request, redirect, url_for, send_file, session, flash
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive Agg
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from models import mongo, User
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.secret_key = '\xfd{H\xe5<\x95\xf9\xe3\x96.5\xd1\x01O<!\xd5\xa2\xa0\x9fR"\xa1\xa8'

# MongoDB configuration
app.config['MONGO_URI'] = 'mongodb+srv://data:123@cluster0.u6ws9.mongodb.net/dataviz_pro?retryWrites=true&w=majority'

# Initialize MongoDB with the app
mongo.init_app(app)

# Add this after mongo.init_app(app) to test the connection
with app.app_context():
    try:
        mongo.db.command('ping')
        print("Successfully connected to MongoDB!")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in first.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def home():
    return render_template('index.html', username=session.get('username'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = mongo.db.users.find_one({'username': username})

        if user and check_password_hash(user['password_hash'], password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
            return render_template('login.html', error='Invalid username or password')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match')
            return render_template('signup.html', error='Passwords do not match')

        if User.get_user_by_username(username):
            flash('Username already exists')
            return render_template('signup.html', error='Username already exists')

        if User.get_user_by_email(email):
            flash('Email already exists')
            return render_template('signup.html', error='Email already exists')

        User.create_user(username, email, password)
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('charts'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('charts'))

    if not file.filename.endswith('.csv'):
        flash('Please upload a CSV file')
        return redirect(url_for('charts'))

    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Generate and store summary
        summary = generate_data_summary(df, session.get('username'))
        
        # Create directory for charts if it doesn't exist
        charts_path = 'static/charts'
        os.makedirs(charts_path, exist_ok=True)

        # Get numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            flash('No numeric columns found in the CSV file')
            return redirect(url_for('charts'))

        # Generate all charts
        chart_urls = {}

        # 1. Bar Chart
        plt.figure(figsize=(12, 6))
        # Group by Product and calculate mean sales
        product_sales = df.groupby('Product')['Sales'].mean().sort_values(ascending=False)
        product_sales.plot(kind='bar')
        plt.title('Average Sales by Product')
        plt.xlabel('Product')
        plt.ylabel('Average Sales')
        plt.xticks(rotation=45)
        plt.tight_layout()
        chart_urls['bar_chart_url'] = save_chart(charts_path, 'bar_chart.png')

        # 2. Line Chart
        plt.figure(figsize=(10, 6))
        df[numeric_columns[0]].plot(kind='line')
        plt.title('Line Chart')
        plt.tight_layout()
        chart_urls['line_chart_url'] = save_chart(charts_path, 'line_chart.png')

        # 3. Pie Chart
        plt.figure(figsize=(10, 6))
        # Take only top 10 values to make pie chart readable
        pie_data = df[numeric_columns[0]].value_counts().head(10)
        plt.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%')
        plt.title('Pie Chart (Top 10 Values)')
        plt.tight_layout()
        chart_urls['pie_chart_url'] = save_chart(charts_path, 'pie_chart.png')

        # 4. Scatter Plot
        plt.figure(figsize=(10, 6))
        if len(numeric_columns) >= 2:
            # Create scatter plot
            plt.scatter(df[numeric_columns[0]], 
                       df[numeric_columns[1]], 
                       alpha=0.6,
                       c='#0d6efd',  # Bootstrap primary color
                       s=100)  # Point size
            
            # Add labels and title
            plt.xlabel(numeric_columns[0])
            plt.ylabel(numeric_columns[1])
            plt.title(f'Scatter Plot: {numeric_columns[0]} vs {numeric_columns[1]}')
            
            # Add grid
            plt.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(df[numeric_columns[0]], df[numeric_columns[1]], 1)
            p = np.poly1d(z)
            plt.plot(df[numeric_columns[0]], 
                    p(df[numeric_columns[0]]), 
                    "r--", 
                    alpha=0.8,
                    label='Trend Line')
            
            # Add legend
            plt.legend()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save with higher DPI for better quality
            chart_urls['scatter_plot_url'] = save_chart(charts_path, 'scatter_plot.png', dpi=300)

        # 5. Bubble Chart
        plt.figure(figsize=(12, 8))
        
        # Get data for bubble chart
        products = df['Product'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(products)))
        
        for product, color in zip(products, colors):
            product_data = df[df['Product'] == product]
            
            # Use Sales for x, Profit for y, and count for size
            x = product_data['Sales']
            y = product_data['Profit']
            size = product_data['Sales'] * 2  # Adjust size multiplier as needed
            
            plt.scatter(x, y, 
                       s=size,  # Size of bubbles
                       alpha=0.6,  # Transparency
                       c=[color],  # Color based on product
                       label=product)  # Add product name to legend
        
        plt.xlabel('Sales')
        plt.ylabel('Profit')
        plt.title('Sales vs Profit by Product\n(Bubble size represents Sales volume)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        chart_urls['bubble_chart_url'] = save_chart(charts_path, 'bubble_chart.png', dpi=300)

        # 6. Area Chart
        plt.figure(figsize=(10, 6))
        df[numeric_columns[:min(3, len(numeric_columns))]].plot(kind='area', alpha=0.5)
        plt.title('Area Chart')
        plt.tight_layout()
        chart_urls['area_chart_url'] = save_chart(charts_path, 'area_chart.png')

        # 7. Histogram
        plt.figure(figsize=(10, 6))
        df[numeric_columns[0]].hist(bins=30)
        plt.title('Histogram')
        plt.tight_layout()
        chart_urls['histogram_url'] = save_chart(charts_path, 'histogram.png')

        # 8. Heatmap
        if len(numeric_columns) >= 2:
            plt.figure(figsize=(10, 6))
            sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm')
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            chart_urls['heatmap_url'] = save_chart(charts_path, 'heatmap.png')

        # 9. Box Plot
        plt.figure(figsize=(10, 6))
        df.boxplot(column=numeric_columns[0])
        plt.title('Box Plot')
        plt.tight_layout()
        chart_urls['boxplot_url'] = save_chart(charts_path, 'boxplot.png')

        # 10. Radar Chart
        if len(df) >= 5:
            plt.figure(figsize=(10, 6))
            values = df[numeric_columns[0]].head(5).values
            categories = [f'Category {i+1}' for i in range(5)]
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
            
            # Close the plot by appending the first value
            values = np.concatenate((values, [values[0]]))
            angles = np.concatenate((angles, [angles[0]]))
            
            ax = plt.subplot(111, projection='polar')
            ax.plot(angles, values)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            plt.title('Radar Chart')
            chart_urls['radar_chart_url'] = save_chart(charts_path, 'radar_chart.png')

        return render_template('charts.html',
                             charts_generated=True,
                             **chart_urls)

    except Exception as e:
        flash(f'Error processing file: {str(e)}')
        return redirect(url_for('charts'))

def save_chart(path, filename, dpi=100):
    """Save the chart with specified DPI"""
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
    return f'/{path}/{filename}'

@app.route('/train_regression', methods=['POST'])
def train_regression():
    file_path = request.form['file_path']
    df = pd.read_csv(file_path)

    input_features = request.form['input_features'].split(',')
    target_feature = request.form['target_feature']

    try:
        X = df[input_features].values
        y = df[target_feature].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Plot actual vs predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color="purple")
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted")
        regression_plot_path = os.path.join(UPLOAD_FOLDER, 'regression.png')
        plt.savefig(regression_plot_path)
        plt.close()

        return render_template('regression_results.html', mse=mse, r2=r2, regression_plot_path=regression_plot_path)

    except Exception as e:
        return f"Error training regression model: {e}"

@app.route('/charts')
@login_required
def charts():
    return render_template('charts.html', username=session.get('username'))

@app.route('/analysis')
@login_required
def analysis():
    return render_template('analysis.html', username=session.get('username'))

@app.route('/summary')
@login_required
def view_summary():
    # Get the latest summary for the current user
    summary = mongo.db.summaries.find_one(
        {'username': session.get('username')},
        sort=[('created_at', -1)]
    )
    return render_template('summary.html', summary=summary)

def generate_data_summary(df, username):
    """Generate summary statistics for the dataframe and store in MongoDB"""
    summary = {
        'username': username,
        'created_at': datetime.utcnow(),
        'total_records': len(df),
        'total_columns': len(df.columns),
        'columns': list(df.columns),
        'numerical_summary': {},
        'categorical_summary': {}
    }

    # Numerical summary
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        summary['numerical_summary'][col] = {
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max())
        }

    # Categorical summary
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_columns:
        value_counts = df[col].value_counts()
        total = len(df)
        summary['categorical_summary'][col] = [
            (str(value), int(count), float(count/total*100))
            for value, count in value_counts.items()
        ]

    # Store in MongoDB
    mongo.db.summaries.insert_one(summary)
    return summary

@app.route('/profile')
@login_required
def profile():
    # Get user data
    user = mongo.db.users.find_one({'username': session.get('username')})
    
    # Get user statistics
    stats = {
        'total_charts': mongo.db.charts.count_documents({'username': session.get('username')}),
        'total_datasets': mongo.db.summaries.count_documents({'username': session.get('username')})
    }
    
    # Get or create recent activities
    recent_activities = list(mongo.db.activities.find(
        {'username': session.get('username')}
    ).sort('timestamp', -1).limit(5))

    # If no activities exist, create some default ones
    if not recent_activities:
        recent_activities = [
            {
                'icon': 'fa-user',
                'title': 'Account Created',
                'description': 'Welcome to DataViz Pro!',
                'timestamp': user.get('created_at', datetime.utcnow())
            }
        ]

    return render_template('profile.html', 
                         user=user, 
                         stats=stats, 
                         recent_activities=recent_activities)

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    
    # Validate inputs
    if not username or not email:
        flash('Username and email are required!', 'error')
        return redirect(url_for('profile'))
    
    # Check if username already exists (if changed)
    if username != session.get('username'):
        existing_user = mongo.db.users.find_one({'username': username})
        if existing_user:
            flash('Username already exists!', 'error')
            return redirect(url_for('profile'))
    
    # Prepare update data
    update_data = {
        'username': username,
        'email': email,
        'updated_at': datetime.utcnow()
    }
    
    # Add password to update if provided
    if password:
        update_data['password'] = generate_password_hash(password)
    
    # Update user in database
    result = mongo.db.users.update_one(
        {'username': session.get('username')},
        {'$set': update_data}
    )
    
    if result.modified_count > 0:
        # Update session if username changed
        if username != session.get('username'):
            session['username'] = username
        
        # Add activity
        mongo.db.activities.insert_one({
            'username': username,
            'icon': 'fa-edit',
            'title': 'Profile Updated',
            'description': 'Profile information was updated successfully',
            'timestamp': datetime.utcnow()
        })
        
        flash('Profile updated successfully!', 'success')
    else:
        flash('No changes made to profile.', 'info')
    
    return redirect(url_for('profile'))

# Add activity tracking function
def track_activity(username, title, description, icon='fa-chart-bar'):
    mongo.db.activities.insert_one({
        'username': username,
        'icon': icon,
        'title': title,
        'description': description,
        'timestamp': datetime.utcnow()
    })

if __name__ == '__main__':
    app.run(debug=True)