import requests
import time

# Test the Flask app
base_url = 'http://localhost:5000'

def test_route(route, description):
    try:
        response = requests.get(f"{base_url}{route}", timeout=5)
        print(f"✅ {description}: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ {description}: {e}")
        return False

# Wait for app to start
print("Waiting for Flask app to start...")
time.sleep(3)

# Test routes
tests = [
    ('/', 'Main page'),
    ('/user/map', 'User map page'),
    ('/admin/login', 'Admin login page'),
    ('/api/predictions', 'Predictions API'),
]

success_count = 0
for route, description in tests:
    if test_route(route, description):
        success_count += 1

print(f"\nTest Results: {success_count}/{len(tests)} routes accessible")

if success_count == len(tests):
    print("🎉 All basic routes are working!")
else:
    print("⚠️ Some routes may have issues.")
