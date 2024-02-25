# models.py
# from database import db
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime



class InRecord(db.Model):
    __tablename__ = 'in_record'
    id = db.Column(db.Integer, primary_key=True)
    license_plate = db.Column(db.String(64), nullable=False)
    image_data = db.Column(db.LargeBinary, nullable=False)  # 或使用 db.BLOB 根据实际数据库适配
    entry_time = db.Column(db.DateTime, default=datetime.utcnow)
    exit_time = db.Column(db.DateTime, default=None, nullable=True)  # 新增離場時間欄位
    ecpay_url = db.Column(db.String, default=None, nullable=True)  # 新增訂單URL欄位
    linepay_url = db.Column(db.String, default=None, nullable=True)  # 新增訂單URL欄位

    def __repr__(self):
        return f'<InRecord {self.license_plate}>'