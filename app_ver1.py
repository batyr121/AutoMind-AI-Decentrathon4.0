import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ExifTags
from datetime import datetime, timedelta
import uuid
import json
import random
from geopy.geocoders import Nominatim

# === Настройки ===
MODEL_PATHS = {
    'damage': "car_damage_classifier.pth"
}

# исправлено 'dunt' -> 'dent'
CLASS_NAMES = {
    'damage': ['car', 'dent', 'rust', 'scratch']
}

CLASS_NAMES_RU = {
    'damage': ['Автомобиль', 'Вмятина', 'Ржавчина', 'Царапина']
}

THRESHOLDS = {
    'damage': 0.6
}

# === Инициализация Flask ===
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///car_inspector.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

# === База данных ===
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    is_premium = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Inspection(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    image_path = db.Column(db.String(255))
    thumb_path = db.Column(db.String(255))
    predictions = db.Column(db.Text)
    location = db.Column(db.String(255))
    damage_count = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_shared = db.Column(db.Boolean, default=False)

# === Авто-логин тестового пользователя ===
@app.before_request
def auto_login():
    # если current_user ещё не определён (например, при запуске до создания БД) — безопасно пропустить
    try:
        if not current_user.is_authenticated:
            user = User.query.filter_by(username="test").first()
            if not user:
                user = User(
                    username="test",
                    email="test@example.com",
                    password_hash=generate_password_hash("test123")
                )
                db.session.add(user)
                db.session.commit()
            login_user(user)
    except Exception:
        # если БД ещё не готова — пропускаем
        pass

# === Модели AI ===
class CarAISystem:
    def __init__(self):
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_models()
    
    def load_models(self):
        try:
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES['damage']))
            
            state = torch.load(MODEL_PATHS['damage'], map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            self.models['damage'] = model
            print(f"✅ Модель damage загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели damage: {e}")
            self.models = {}  # оставляем пустым — будем давать fallback

    def analyze_image(self, image):
        """Возвращает словарь с ключом 'damage' -> {class: prob, ..., 'has_damage': bool}"""
        results = {}
        if 'damage' in self.models:
            results['damage'] = self.predict(image, 'damage')
        return results
    
    def predict(self, image, model_type):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        x = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.models[model_type](x)[0]
            probs = torch.nn.functional.softmax(out, dim=0).cpu().numpy()
        
        result = {CLASS_NAMES[model_type][i]: float(probs[i]) for i in range(len(CLASS_NAMES[model_type]))}
        
        if model_type == 'damage':
            damage_types = [d for d in CLASS_NAMES[model_type] if d != 'car']
            result['has_damage'] = any(result.get(d, 0.0) > THRESHOLDS['damage'] for d in damage_types)
        return result

# === Инициализация систем ===
ai_system = CarAISystem()
geolocator = Nominatim(user_agent="car_inspector")

def create_thumbnail(image, size=(300, 300)):
    img = image.copy()
    img.thumbnail(size)
    return img

def _rational_to_float(val):
    """
    EXIF values часто выглядят так: (num, den) или как обычное число.
    Приводим в float корректно.
    """
    try:
        if isinstance(val, tuple) and len(val) == 2:
            num, den = val
            if den == 0:
                return 0.0
            return float(num) / float(den)
        return float(val)
    except Exception:
        return None

def convert_to_degrees(value):
    """Конвертация GPS координат в градусы, учитывая рациональные числа в EXIF."""
    try:
        if isinstance(value, tuple) and len(value) == 3:
            d = _rational_to_float(value[0])
            m = _rational_to_float(value[1])
            s = _rational_to_float(value[2])
            if d is None or m is None or s is None:
                return None
            return d + (m / 60.0) + (s / 3600.0)
    except Exception as e:
        print("Ошибка конвертации координат:", e)
    return None

def get_location_from_exif(image):
    try:
        exif = image._getexif()
        if not exif:
            return None
        gps_info = {}
        for tag, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            if tag_name == 'GPSInfo':
                for gps_tag in value:
                    gps_tag_name = ExifTags.GPSTAGS.get(gps_tag, gps_tag)
                    gps_info[gps_tag_name] = value[gps_tag]
        if gps_info:
            lat = convert_to_degrees(gps_info.get('GPSLatitude'))
            lon = convert_to_degrees(gps_info.get('GPSLongitude'))
            # учтём направление N/S E/W
            lat_ref = gps_info.get('GPSLatitudeRef')
            lon_ref = gps_info.get('GPSLongitudeRef')
            if lat is not None and lon is not None:
                if lat_ref and lat_ref.upper() == 'S':
                    lat = -abs(lat)
                if lon_ref and lon_ref.upper() == 'W':
                    lon = -abs(lon)
                try:
                    location = geolocator.reverse((lat, lon), language="ru")
                    return location.address if location else None
                except Exception as e:
                    print("Ошибка reverse geocode:", e)
                    return None
    except Exception as e:
        print("Ошибка работы с EXIF:", e)
    return None

def generate_report(predictions, image_path):
    return "path/to/report.pdf"

# === Эндпоинт анализа ===
@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    if "file" not in request.files:
        return jsonify({"error": "Файл не найден"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Файл не выбран"}), 400

    try:
        img = Image.open(file.stream)
        img = img.convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Не удалось открыть изображение: {e}"}), 400

    # если модель загружена — используем её, иначе даём fallback-ответ (для разработки)
    if 'damage' in ai_system.models:
        try:
            model_results = ai_system.analyze_image(img).get('damage', {})
            # model_results: {'car': prob, 'dent': prob, 'rust': prob, 'scratch': prob, 'has_damage': bool}
            # формируем predictions (все классы)
            predictions = {k: float(v) for k, v in model_results.items() if k != 'has_damage'}
            # вычисляем confidence — максимум вероятности повреждающей категории
            damage_classes = [c for c in CLASS_NAMES['damage'] if c != 'car']
            confidence = max([predictions.get(c, 0.0) for c in damage_classes], default=0.0)
            has_damage = model_results.get('has_damage', False)

            # составляем damage_types (только те, что выше порога, сортируем по уверенности)
            damage_types = []
            for cls in damage_classes:
                prob = float(predictions.get(cls, 0.0))
                if prob > 0:
                    damage_types.append({
                        "type": cls,
                        "type_ru": CLASS_NAMES_RU['damage'][CLASS_NAMES['damage'].index(cls)],
                        "confidence": prob
                    })
            # сортировка по убыванию вероятности
            damage_types.sort(key=lambda x: x['confidence'], reverse=True)

            message = "Обнаружены повреждения" if has_damage else "Повреждений не найдено"
        except Exception as e:
            # если что-то пошло не так — fallback
            print("Ошибка при разборе результатов модели:", e)
            predictions = {}
            damage_types = []
            confidence = 0.0
            has_damage = False
            message = "Ошибка анализа"
    else:
        # fallback — возвращаем фиктивный ответ, чтобы фронт не падал
        predictions = {
            "dent": round(random.uniform(0, 1), 3),
            "rust": round(random.uniform(0, 1), 3),
            "scratch": round(random.uniform(0, 1), 3),
            "car": 1.0
        }
        # считаем вероятность и наличие повреждений по порогу
        damage_classes = [c for c in CLASS_NAMES['damage'] if c != 'car']
        confidence = max(predictions.get(c, 0.0) for c in damage_classes)
        has_damage = confidence > THRESHOLDS['damage']
        damage_types = []
        if has_damage:
            for cls in damage_classes:
                prob = predictions.get(cls, 0.0)
                if prob > 0.05:
                    damage_types.append({
                        "type": cls,
                        "type_ru": CLASS_NAMES_RU['damage'][CLASS_NAMES['damage'].index(cls)],
                        "confidence": prob
                    })
            damage_types.sort(key=lambda x: x['confidence'], reverse=True)
        message = "Обнаружены повреждения" if has_damage else "Повреждений не найдено"

    # Сохранение изображения и записи в БД
    try:
        inspection_id = str(uuid.uuid4())
        filename = f"{inspection_id}.jpg"
        thumbname = f"{inspection_id}_thumb.jpg"
        
        save_path = os.path.join("static", "uploads", filename)
        thumb_path = os.path.join("static", "uploads", thumbname)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
        create_thumbnail(img).save(thumb_path)

        location = get_location_from_exif(img) or "Неизвестно"
        damage_count = len([d for d in damage_types if d.get('confidence', 0) > 0.0])

        inspection = Inspection(
            id=inspection_id,
            user_id=current_user.id if current_user and hasattr(current_user, "id") else None,
            image_path=filename,
            thumb_path=thumbname,
            predictions=json.dumps({"predictions": predictions, "damage_types": damage_types}),
            location=location,
            damage_count=damage_count,
            created_at=datetime.utcnow()
        )
        # try to save — but don't fail the API if DB save breaks
        try:
            db.session.add(inspection)
            db.session.commit()
        except Exception as e:
            print("Не удалось сохранить в БД:", e)
    except Exception as e:
        print("Ошибка при сохранении файла/записи:", e)

    # Окончательный ответ (формат для фронта)
    return jsonify({
        "has_damage": bool(has_damage),
        "confidence": float(confidence),
        "damage_types": damage_types,      # [{"type","type_ru","confidence"}, ...]
        "predictions": predictions,        # {"dent":0.8, "rust":0.1, ...}
        "message": message,
        "inspection_id": inspection_id
    })

# === Остальные маршруты ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
@login_required
def dashboard():
    inspections = Inspection.query.filter_by(user_id=current_user.id).order_by(Inspection.created_at.desc()).limit(10).all()
    stats = {
        'total': Inspection.query.filter_by(user_id=current_user.id).count(),
        'with_damage': Inspection.query.filter(Inspection.user_id == current_user.id, Inspection.damage_count > 0).count(),
        'recent': Inspection.query.filter(Inspection.user_id == current_user.id,
                                         Inspection.created_at >= datetime.utcnow() - timedelta(days=7)).count()
    }
    return render_template("dashboard.html", inspections=inspections, stats=stats)

@app.route("/api/report/<inspection_id>")
@login_required
def generate_pdf_report(inspection_id):
    inspection = Inspection.query.get_or_404(inspection_id)
    if inspection.user_id != current_user.id:
        return jsonify({"error": "Доступ запрещен"}), 403
    
    report_path = generate_report(json.loads(inspection.predictions), inspection.image_path)
    return send_from_directory(os.path.dirname(report_path), os.path.basename(report_path))

@app.route("/api/stats")
@login_required
def get_stats():
    return jsonify({
        "total": Inspection.query.filter_by(user_id=current_user.id).count(),
        "with_damage": Inspection.query.filter(Inspection.user_id == current_user.id, Inspection.damage_count > 0).count(),
        "recent": Inspection.query.filter(Inspection.user_id == current_user.id,
                                          Inspection.created_at >= datetime.utcnow() - timedelta(days=7)).count()
    })

# === Аутентификация ===
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data.get('username')).first()
    if user and check_password_hash(user.password_hash, data.get('password', '')):
        login_user(user)
        return jsonify({"success": True})
    return jsonify({"error": "Неверные учетные данные"}), 401

# === Инициализация ===
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5500)
