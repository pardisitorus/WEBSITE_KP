# TODO List for Forest Fire Prediction Website

## 1. Setup Project Structure
- [x] Create directories: templates, static, models, data
- [x] Move existing files to appropriate directories (MODEL.py to models/, data files to data/)

## 2. Install Dependencies
- [x] Install Python packages: Flask, pandas, scikit-learn, geopandas, shapely, folium, joblib

## 3. Data Preparation
- [x] Convert desa1_riau.csv to GeoJSON format for map visualization
- [x] Prepare village data with predictions

## 4. Backend Development (Flask App)
- [x] Create app.py with main Flask application
- [x] Implement admin login route with password "admin123"
- [x] Create admin dashboard for model monitoring
- [x] Implement model upload functionality with metadata
- [x] Add canary deployment logic (traffic shifting between old and new models)
- [x] Create user routes for map visualization
- [x] Implement prediction API for live predictions

## 5. Frontend Development
- [x] Create HTML templates for admin login, dashboard, user map
- [x] Add CSS styling for responsive design
- [x] Integrate Leaflet.js for interactive map
- [x] Implement JavaScript for map interactions (village details, risk levels)
- [x] Add educational menu with prevention recommendations

## 6. Model Integration
- [x] Refactor MODEL.py into reusable functions
- [x] Implement model loading and prediction functions
- [x] Add model versioning and storage in models/ directory

## 7. Testing and Deployment
- [x] Ensure website runs locally without errors
- [ ] Test admin login and dashboard functionality
- [ ] Test model upload and canary deployment
- [ ] Test user map visualization and interactions
- [ ] Validate predictions and data accuracy
