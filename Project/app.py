from flask import Flask, render_template, request, redirect, session, flash
import pickle, sqlite3, pandas as pd, os
from datetime import timedelta

app = Flask(__name__)

# ─── KONFIGURASI ───────────────────────────────────────────────────────────────
# PENTING: Ganti secret_key dengan string acak yang kuat sebelum deploy!
# Contoh generate: python -c "import secrets; print(secrets.token_hex(32))"
app.secret_key = os.environ.get('SECRET_KEY', 'ganti-dengan-key-rahasia-yang-kuat')
app.permanent_session_lifetime = timedelta(minutes=30)

# Kredensial admin — idealnya simpan di environment variable
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', '123')

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
try:
    model  = pickle.load(open('model_depression_balanced.pkl', 'rb'))
    scaler = pickle.load(open('scaler_balanced.pkl', 'rb'))
except FileNotFoundError as e:
    raise RuntimeError(f"File model tidak ditemukan: {e}. Pastikan file .pkl ada di direktori yang sama dengan app.py.")

# ─── DATABASE ──────────────────────────────────────────────────────────────────
def get_db():
    """Buat koneksi SQLite baru. Selalu tutup setelah dipakai."""
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row  # Akses kolom by name: row['age']
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                age        REAL,
                cgpa       REAL,
                study      REAL,
                sleep      REAL,
                social     REAL,
                activity   REAL,
                stress     REAL,
                gender     TEXT,
                department TEXT,
                result     TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

init_db()

# ─── HELPER: VALIDASI INPUT ────────────────────────────────────────────────────
def validate_predict_input(form):
    """
    Validasi dan parse input form prediksi.
    Return (data_dict, error_message).
    error_message = None jika valid.
    """
    errors = []

    def parse_float(field, label, min_val, max_val):
        try:
            val = float(form[field])
            if not (min_val <= val <= max_val):
                errors.append(f"{label} harus antara {min_val}–{max_val}.")
            return val
        except (ValueError, KeyError):
            errors.append(f"{label} tidak valid.")
            return None

    age      = parse_float('age',      'Usia',            15,  60)
    cgpa     = parse_float('cgpa',     'IPK',             0.0, 4.0)
    study    = parse_float('study',    'Jam Belajar',     0,   24)
    sleep    = parse_float('sleep',    'Jam Tidur',       0,   24)
    social   = parse_float('social',   'Jam Sosmed',      0,   24)
    activity = parse_float('activity', 'Aktivitas Fisik', 0,   600)
    stress   = parse_float('stress',   'Tingkat Stres',   1,   10)

    gender     = form.get('gender', '')
    department = form.get('department', '')

    if gender not in ('Male', 'Female'):
        errors.append("Pilih jenis kelamin yang valid.")
    if department not in ('Engineering', 'Business', 'Arts', 'Medical', 'Science'):
        errors.append("Pilih departemen yang valid.")

    if errors:
        return None, ' '.join(errors)

    return {
        'age': age, 'cgpa': cgpa, 'study': study, 'sleep': sleep,
        'social': social, 'activity': activity, 'stress': stress,
        'gender': gender, 'department': department
    }, None

# ─── HELPER: INSIGHT & INTERPRETASI ───────────────────────────────────────────
def build_insight(data):
    insight = []

    # Stres (skala 1–10: 1–3 Rendah, 4–6 Sedang, 7–10 Tinggi)
    if data['stress'] >= 7:
        insight.append("Tingkat stres tinggi — pertimbangkan teknik relaksasi, meditasi, atau konseling profesional.")
    elif data['stress'] >= 4:
        insight.append("Tingkat stres sedang — kelola dengan olahraga, meditasi, atau aktivitas yang menenangkan.")
    else:
        insight.append("Tingkat stres rendah — pertahankan keseimbangan dan kelola tekanan secara rutin.")

    # Tidur
    if data['sleep'] < 7:
        insight.append("Durasi tidur kurang — tidur di bawah 7 jam dapat meningkatkan risiko gangguan mood.")
    elif data['sleep'] > 9:
        insight.append("Tidur lebih dari 9 jam — bisa berdampak pada ritme tubuh, pertahankan pola tidur seimbang.")
    else:
        insight.append("Durasi tidur optimal — pertahankan konsistensi tidur 7–9 jam per hari.")

    # Media sosial
    if data['social'] > 3:
        insight.append("Penggunaan media sosial tinggi — waktu layar berlebih dapat berkorelasi dengan kecemasan.")
    else:
        insight.append("Penggunaan media sosial wajar — relatif aman dan tidak menunjukkan dampak negatif signifikan.")

    # Aktivitas fisik
    if data['activity'] < 30:
        insight.append("Aktivitas fisik rendah — olahraga ringan minimal 30 menit per hari dapat mendukung kesehatan mental.")
    elif data['activity'] > 60:
        insight.append("Aktivitas fisik tinggi — pastikan tidak mengganggu aktivitas lain atau menyebabkan kelelahan.")
    else:
        insight.append("Aktivitas fisik optimal — pertahankan konsistensi dan keseimbangan dengan kegiatan lain.")

    # Prestasi akademik
    if data['cgpa'] < 2.75:
        insight.append("Performa akademik rendah — pertimbangkan bimbingan akademik atau strategi belajar yang lebih efektif.")

    # Jam belajar
    if data['study'] > 2:
        insight.append("Jam belajar tinggi — efektivitas dapat menurun, sisipkan waktu istirahat.")
    elif data['study'] < 1:
        insight.append("Jam belajar rendah — pertimbangkan peningkatan durasi belajar secara bertahap.")
    else:
        insight.append("Jam belajar optimal — pertahankan konsistensi dan fokus belajar.")

    # Tidak ada peringatan khusus
    if not insight:
        insight.append("Gaya hidup Anda relatif seimbang — pertahankan pola ini dan tetap perhatikan kondisi mental.")

    return insight

def build_interpretasi(data):
    def status_sleep(v):
        if 7 <= v <= 9: return "Normal"
        return "Rendah" if v < 7 else "Tinggi"

    def status_stress(v):
        # Skala 1–10: 1–3 Rendah, 4–6 Sedang, 7–10 Tinggi
        if v <= 3: return "Rendah"
        if v <= 6: return "Sedang"
        return "Tinggi"

    def status_activity(v):
        if 30 <= v <= 60: return "Normal"
        return "Rendah" if v < 60 else "Tinggi"

    def status_social(v):
        # Threshold diperbaiki sesuai build_insight: normal < 3 jam
        return "Normal" if v < 3 else "Tinggi"

    def status_cgpa(v):
        return "Normal" if v >= 3.0 else "Rendah"

    def status_study(v):
        # Normal: 1–2 jam sesuai logika build_insight
        if 1 <= v <= 2: return "Normal"
        return "Rendah" if v < 1 else "Tinggi"

    return {
        "IPK":          {"value": data['cgpa'],     "normal": "≥ 3.0",        "status": status_cgpa(data['cgpa'])},
        "Belajar":      {"value": data['study'],    "normal": "1–2 jam",      "status": status_study(data['study'])},
        "Tidur":        {"value": data['sleep'],    "normal": "7–9 jam",      "status": status_sleep(data['sleep'])},
        "Media Sosial": {"value": data['social'],   "normal": "< 3 jam",      "status": status_social(data['social'])},
        "Aktivitas":    {"value": data['activity'], "normal": "30–60 menit",  "status": status_activity(data['activity'])},
        "Stres":        {"value": data['stress'],   "normal": "1–3",          "status": status_stress(data['stress'])},
    }

# ─── ROUTES ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/overview')
def overview():
    with get_db() as conn:
        c = conn.cursor()

        depresi    = c.execute("SELECT COUNT(*) FROM predictions WHERE result='Depresi'").fetchone()[0]
        tidak      = c.execute("SELECT COUNT(*) FROM predictions WHERE result='Tidak Depresi'").fetchone()[0]
        avg_stress = round(c.execute("SELECT AVG(stress) FROM predictions").fetchone()[0] or 0, 2)
        avg_sleep  = round(c.execute("SELECT AVG(sleep)  FROM predictions").fetchone()[0] or 0, 2)

        cgpa_vs_dep     = c.execute("SELECT result, ROUND(AVG(cgpa),2)     FROM predictions GROUP BY result").fetchall()
        sleep_vs_dep    = c.execute("SELECT result, ROUND(AVG(sleep),2)    FROM predictions GROUP BY result").fetchall()
        study_vs_dep    = c.execute("SELECT result, ROUND(AVG(study),2)    FROM predictions GROUP BY result").fetchall()
        social_vs_dep   = c.execute("SELECT result, ROUND(AVG(social),2)   FROM predictions GROUP BY result").fetchall()
        activity_vs_dep = c.execute("SELECT result, ROUND(AVG(activity),2) FROM predictions GROUP BY result").fetchall()
        stress_vs_dep   = c.execute("SELECT result, ROUND(AVG(stress),2)   FROM predictions GROUP BY result").fetchall()

        # Helper: konversi sqlite3.Row → list biasa agar bisa di-tojson di template
        def rows(sql):
            return [list(r) for r in c.execute(sql).fetchall()]

        # ── Distribusi per Gender & Department ──────────────────────────────
        gender_dep = rows("SELECT gender, result, COUNT(*) FROM predictions GROUP BY gender, result ORDER BY gender, result")
        dept_dep   = rows("SELECT department, result, COUNT(*) FROM predictions GROUP BY department, result ORDER BY department, result")

        # ── Analisis 3 Variabel: (variabel, gender, depression) ─────────────
        cgpa_gender_dep     = rows("SELECT gender, result, ROUND(AVG(cgpa),2)     FROM predictions GROUP BY gender, result ORDER BY gender, result")
        sleep_gender_dep    = rows("SELECT gender, result, ROUND(AVG(sleep),2)    FROM predictions GROUP BY gender, result ORDER BY gender, result")
        study_gender_dep    = rows("SELECT gender, result, ROUND(AVG(study),2)    FROM predictions GROUP BY gender, result ORDER BY gender, result")
        social_gender_dep   = rows("SELECT gender, result, ROUND(AVG(social),2)   FROM predictions GROUP BY gender, result ORDER BY gender, result")
        activity_gender_dep = rows("SELECT gender, result, ROUND(AVG(activity),2) FROM predictions GROUP BY gender, result ORDER BY gender, result")
        stress_gender_dep   = rows("SELECT gender, result, ROUND(AVG(stress),2)   FROM predictions GROUP BY gender, result ORDER BY gender, result")

        # ── Analisis 3 Variabel: (variabel, department, depression) ─────────
        cgpa_dept_dep     = rows("SELECT department, result, ROUND(AVG(cgpa),2)     FROM predictions GROUP BY department, result ORDER BY department, result")
        sleep_dept_dep    = rows("SELECT department, result, ROUND(AVG(sleep),2)    FROM predictions GROUP BY department, result ORDER BY department, result")
        study_dept_dep    = rows("SELECT department, result, ROUND(AVG(study),2)    FROM predictions GROUP BY department, result ORDER BY department, result")
        social_dept_dep   = rows("SELECT department, result, ROUND(AVG(social),2)   FROM predictions GROUP BY department, result ORDER BY department, result")
        activity_dept_dep = rows("SELECT department, result, ROUND(AVG(activity),2) FROM predictions GROUP BY department, result ORDER BY department, result")
        stress_dept_dep   = rows("SELECT department, result, ROUND(AVG(stress),2)   FROM predictions GROUP BY department, result ORDER BY department, result")

    return render_template('overview.html',
        depresi=depresi, tidak=tidak,
        avg_stress=avg_stress, avg_sleep=avg_sleep,
        sleep_vs_dep=sleep_vs_dep, stress_vs_dep=stress_vs_dep,
        activity_vs_dep=activity_vs_dep, cgpa_vs_dep=cgpa_vs_dep,
        study_vs_dep=study_vs_dep, social_vs_dep=social_vs_dep,
        # Gender & Department distribution
        gender_dep=gender_dep, dept_dep=dept_dep,
        # 3-var: variabel + gender + depression
        cgpa_gender_dep=cgpa_gender_dep, sleep_gender_dep=sleep_gender_dep,
        study_gender_dep=study_gender_dep, social_gender_dep=social_gender_dep,
        activity_gender_dep=activity_gender_dep, stress_gender_dep=stress_gender_dep,
        # 3-var: variabel + department + depression
        cgpa_dept_dep=cgpa_dept_dep, sleep_dept_dep=sleep_dept_dep,
        study_dept_dep=study_dept_dep, social_dept_dep=social_dept_dep,
        activity_dept_dep=activity_dept_dep, stress_dept_dep=stress_dept_dep,
    )


@app.route('/predict')
def predict_page():
    return render_template('predict.html')


@app.route('/predict_result', methods=['POST'])
def predict():
    session.permanent = False

    # Validasi input
    data, error = validate_predict_input(request.form)
    if error:
        flash(error, 'danger')
        return redirect('/predict')

    # Buat DataFrame untuk prediksi
    input_dict = {
        'Age':                data['age'],
        'CGPA':               data['cgpa'],
        'Sleep_Duration':     data['sleep'],
        'Study_Hours':        data['study'],
        'Social_Media_Hours': data['social'],
        'Physical_Activity':  data['activity'],
        'Stress_Level':       data['stress'],
        'Gender_Female':      1 if data['gender'] == 'Female' else 0,
        'Gender_Male':        1 if data['gender'] == 'Male'   else 0,
        'Department_Arts':        1 if data['department'] == 'Arts'        else 0,
        'Department_Business':    1 if data['department'] == 'Business'    else 0,
        'Department_Engineering': 1 if data['department'] == 'Engineering' else 0,
        'Department_Medical':     1 if data['department'] == 'Medical'     else 0,
        'Department_Science':     1 if data['department'] == 'Science'     else 0,
    }

    df    = pd.DataFrame([input_dict])
    final = scaler.transform(df)

    pred  = model.predict(final)[0]
    prob  = round(float(model.predict_proba(final)[0][1]) * 100, 2)

    result = "Depresi" if pred == 1 else "Tidak Depresi"
    level  = "Tinggi" if prob > 75 else ("Sedang" if prob >= 50 else "Rendah")

    # Simpan ke sesi
    session['result']       = result
    session['prob']         = prob
    session['level']        = level
    session['insight']      = build_insight(data)
    session['interpretasi'] = build_interpretasi(data)

    # Simpan ke database
    with get_db() as conn:
        conn.execute(
            "INSERT INTO predictions (age,cgpa,study,sleep,social,activity,stress,gender,department,result) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (data['age'], data['cgpa'], data['study'], data['sleep'],
             data['social'], data['activity'], data['stress'],
             data['gender'], data['department'], result)
        )
        conn.commit()

    return redirect('/result')


@app.route('/result')
def result_page():
    if 'result' not in session:
        return redirect('/predict')
    return render_template('result.html',
        result=session['result'],
        prob=session['prob'],
        level=session['level'],
        insight=session['insight'],
        interpretasi=session['interpretasi'],
    )


@app.route('/clear')
def clear():
    session.clear()
    return redirect('/')


# ─── ADMIN ─────────────────────────────────────────────────────────────────────

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if session.get('admin'):
        return redirect('/admin')
    error = None
    if request.method == 'POST':
        if (request.form.get('username') == ADMIN_USERNAME and
                request.form.get('password') == ADMIN_PASSWORD):
            session.permanent = True
            session['admin'] = True
            return redirect('/admin')
        error = "Username atau password salah."
    return render_template('admin_login.html', error=error)


@app.route('/admin')
def admin_dashboard():
    if not session.get('admin'):
        return redirect('/admin/login')
    with get_db() as conn:
        data = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    return render_template('admin_dashboard.html', data=data)


@app.route('/admin/delete/<int:id>')
def delete(id):
    if not session.get('admin'):
        return redirect('/admin/login')
    with get_db() as conn:
        conn.execute("DELETE FROM predictions WHERE id=?", (id,))
        conn.commit()
    return redirect('/admin')


@app.route('/admin/logout')
def logout():
    session.pop('admin', None)
    return redirect('/')


# ─── ENTRYPOINT ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # debug=True hanya untuk development — matikan saat deploy ke production
    app.run(debug=True)