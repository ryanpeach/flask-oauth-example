from app import app, db, lm
import os

if __name__ == '__main__':
    db.create_all()
    app.run(host=os.getenv('IP', '0.0.0.0'), port=int(os.getenv('PORT', 8080)))
