import sys
sys.path.append("src")
sys.path.append("src/app")
from app import app

application = app.App()
application.run()