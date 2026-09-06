from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib

app = FastAPI(
    title="Student Failure Risk Prediction API",
    description="Система раннего предупреждения академической неуспеваемости (Уровень 2)",
    version="1.0.0"
)

models = joblib.load('models_l2.joblib')
calibrator = joblib.load('calibrator_l2.joblib')
OPTIMAL_THRESHOLD = 0.1304


class StudentData(BaseModel):
    school: str = Field(default="GP")
    sex: str = Field(default="F")
    age: int = Field(default=16, ge=15, le=22)
    address: str = Field(default="U")
    famsize: str = Field(default="GT3")
    Pstatus: str = Field(default="T")
    Medu: int = Field(default=2)
    Fedu: int = Field(default=2)
    Mjob: str = Field(default="services")
    Fjob: str = Field(default="other")
    reason: str = Field(default="course")
    guardian: str = Field(default="mother")
    traveltime: int = Field(default=1)
    studytime: int = Field(default=2)
    failures: int = Field(default=0, ge=0, le=3)
    schoolsup: str = Field(default="no")
    famsup: str = Field(default="yes")
    paid: str = Field(default="no")
    activities: str = Field(default="no")
    nursery: str = Field(default="yes")
    higher: str = Field(default="yes")
    internet: str = Field(default="yes")
    romantic: str = Field(default="no")
    famrel: int = Field(default=4)
    freetime: int = Field(default=3)
    goout: int = Field(default=3)
    Dalc: int = Field(default=1)
    Walc: int = Field(default=1)
    health: int = Field(default=3)
    absences: int = Field(default=4, ge=0, le=32)


@app.post("/predict")
def predict_risk(student: StudentData):
    input_df = pd.DataFrame([student.model_dump()])

    cat_cols = input_df.select_dtypes(include=['string']).columns
    for c in cat_cols:
        input_df[c] = input_df[c].astype(object)

    raw_prob = np.mean([m.predict_proba(input_df)[:, 1] for m in models], axis=0)
    calibrated_prob = float(calibrator.transform(raw_prob)[0])
    is_at_risk = bool(calibrated_prob >= OPTIMAL_THRESHOLD)

    return {
        "failure_probability": round(calibrated_prob, 4),
        "risk_group": is_at_risk,
        "recommendation": "Срочно пригласить на беседу с куратором" if is_at_risk else "Академических рисков не выявлено"
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Мониторинг академических рисков студентов</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * { box-sizing: border-box; font-family: 'Inter', sans-serif; }
            body { background: #0f172a; margin: 0; padding: 25px 15px; color: #f8fafc; }
            .container { max-width: 1050px; margin: 0 auto; background: #1e293b; border-radius: 20px; border: 1px solid #334155; padding: 35px; box-shadow: 0 20px 40px rgba(0,0,0,0.3); }
            .header { border-bottom: 1px solid #334155; padding-bottom: 20px; margin-bottom: 25px; }
            h1 { margin: 0 0 8px 0; font-size: 26px; color: #f8fafc; }
            p { margin: 0; color: #94a3b8; font-size: 14px; }

            .presets { display: flex; gap: 10px; margin-bottom: 25px; flex-wrap: wrap; }
            .preset-btn { border: none; padding: 9px 16px; border-radius: 8px; font-weight: 600; font-size: 13px; cursor: pointer; transition: opacity 0.2s; }
            .preset-btn:hover { opacity: 0.9; }
            .preset-good { background: #0284c7; color: white; }
            .preset-bad { background: #dc2626; color: white; }

            .section-box { background: #0f172a; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
            .highlight-box { background: #2e1065; border: 1px solid #7c3aed; }
            .section-title { font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: #38bdf8; margin: 0 0 15px 0; }
            .highlight-title { color: #c084fc; }

            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 15px; }
            label { display: block; font-size: 12px; font-weight: 500; margin-bottom: 6px; color: #cbd5e1; }
            input, select { width: 100%; padding: 9px 12px; background: #1e293b; border: 1.5px solid #475569; border-radius: 8px; font-size: 13px; color: #f8fafc; outline: none; transition: border-color 0.2s; }
            input:focus, select:focus { border-color: #38bdf8; }

            .btn-submit { width: 100%; background: #6366f1; color: white; padding: 15px; border: none; border-radius: 10px; font-size: 16px; font-weight: 700; cursor: pointer; margin-top: 15px; transition: background 0.2s; }
            .btn-submit:hover { background: #4f46e5; }

            .result-card { margin-top: 25px; padding: 22px; border-radius: 14px; display: none; }
            .result-safe { background: #064e3b; border: 1.5px solid #10b981; color: #ecfdf5; }
            .result-danger { background: #7f1d1d; border: 1.5px solid #ef4444; color: #fef2f2; }
            .prob-value { font-size: 38px; font-weight: 800; margin: 6px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Академический риск: Панель предиктивного скоринга</h1>
                <p>Ранний прогноз неуспеваемости до проведения промежуточных контрольных (все 30 параметров модели)</p>
            </div>

            <div class="presets">
                <span style="font-size: 13px; font-weight: 600; align-self: center; margin-right: 5px; color: #94a3b8;">Быстрые сценарии:</span>
                <button class="preset-btn preset-good" onclick="setPreset('good')">Благополучный профиль</button>
                <button class="preset-btn preset-bad" onclick="setPreset('risk')">Критический риск</button>
            </div>

            <form id="studentForm">
                <!-- 1. ТОП-3 ФАКТОРА -->
                <div class="section-box highlight-box">
                    <div class="section-title highlight-title">Топ-3 фактора риска (SHAP-анализ)</div>
                    <div class="grid">
                        <div>
                            <label>Учебное заведение (school):</label>
                            <select id="school" name="school">
                                <option value="GP">GP (Gabriel Pereira)</option>
                                <option value="MS">MS (Mousinho da Silveira)</option>
                            </select>
                        </div>
                        <div>
                            <label>Прошлые незачёты (failures):</label>
                            <select id="failures" name="failures">
                                <option value="0">0 незачётов</option>
                                <option value="1">1 незачёт</option>
                                <option value="2">2 незачёта</option>
                                <option value="3">3 незачёта</option>
                            </select>
                        </div>
                        <div>
                            <label>Пропуски занятий (absences):</label>
                            <input type="number" id="absences" name="absences" min="0" max="93" value="4">
                        </div>
                    </div>
                </div>

                <!-- 2. УЧЕБА И ПОДГОТОВКА -->
                <div class="section-box">
                    <div class="section-title">Учебная активность и цели</div>
                    <div class="grid">
                        <div>
                            <label>Самоподготовка (studytime):</label>
                            <select id="studytime" name="studytime">
                                <option value="1">&lt; 2 часов/нед</option>
                                <option value="2" selected>2 - 5 часов/нед</option>
                                <option value="3">5 - 10 часов/нед</option>
                                <option value="4">&gt; 10 часов/нед</option>
                            </select>
                        </div>
                        <div>
                            <label>Время на дорогу (traveltime):</label>
                            <select id="traveltime" name="traveltime">
                                <option value="1">&lt; 15 мин</option>
                                <option value="2">15 - 30 мин</option>
                                <option value="3">30 - 60 мин</option>
                                <option value="4">&gt; 1 часа</option>
                            </select>
                        </div>
                        <div>
                            <label>Высшее образование (higher):</label>
                            <select id="higher" name="higher">
                                <option value="yes" selected>Планирует</option>
                                <option value="no">Не планирует</option>
                            </select>
                        </div>
                        <div>
                            <label>Поддержка школы (schoolsup):</label>
                            <select id="schoolsup" name="schoolsup">
                                <option value="no" selected>Нет</option>
                                <option value="yes">Да</option>
                            </select>
                        </div>
                        <div>
                            <label>Платные доп. занятия (paid):</label>
                            <select id="paid" name="paid">
                                <option value="no" selected>Нет</option>
                                <option value="yes">Да</option>
                            </select>
                        </div>
                        <div>
                            <label>Причина выбора школы (reason):</label>
                            <select id="reason" name="reason">
                                <option value="home">Близко к дому</option>
                                <option value="reputation">Репутация</option>
                                <option value="course" selected>Учебная программа</option>
                                <option value="other">Другое</option>
                            </select>
                        </div>
                    </div>
                </div>

                <!-- 3. ДЕМОГРАФИЯ И СЕМЬЯ -->
                <div class="section-box">
                    <div class="section-title">Личные данные и семья</div>
                    <div class="grid">
                        <div>
                            <label>Пол (sex):</label>
                            <select id="sex" name="sex">
                                <option value="F">Женский</option>
                                <option value="M">Мужской</option>
                            </select>
                        </div>
                        <div>
                            <label>Возраст (age):</label>
                            <input type="number" id="age" name="age" min="15" max="22" value="16">
                        </div>
                        <div>
                            <label>Тип местности (address):</label>
                            <select id="address" name="address">
                                <option value="U" selected>Город (U)</option>
                                <option value="R">Село (R)</option>
                            </select>
                        </div>
                        <div>
                            <label>Размер семьи (famsize):</label>
                            <select id="famsize" name="famsize">
                                <option value="GT3" selected>&gt; 3 человек</option>
                                <option value="LE3">&le; 3 человек</option>
                            </select>
                        </div>
                        <div>
                            <label>Родители живут вместе (Pstatus):</label>
                            <select id="Pstatus" name="Pstatus">
                                <option value="T" selected>Вместе</option>
                                <option value="A">Раздельно</option>
                            </select>
                        </div>
                        <div>
                            <label>Помощь семьи в учебе (famsup):</label>
                            <select id="famsup" name="famsup">
                                <option value="yes" selected>Да</option>
                                <option value="no">Нет</option>
                            </select>
                        </div>
                        <div>
                            <label>Образование матери (Medu):</label>
                            <select id="Medu" name="Medu">
                                <option value="0">Нет образования</option>
                                <option value="1">Начальное (4 класса)</option>
                                <option value="2" selected>Базовое (5-9 классов)</option>
                                <option value="3">Среднее общее</option>
                                <option value="4">Высшее</option>
                            </select>
                        </div>
                        <div>
                            <label>Образование отца (Fedu):</label>
                            <select id="Fedu" name="Fedu">
                                <option value="0">Нет образования</option>
                                <option value="1">Начальное (4 класса)</option>
                                <option value="2" selected>Базовое (5-9 классов)</option>
                                <option value="3">Среднее общее</option>
                                <option value="4">Высшее</option>
                            </select>
                        </div>
                        <div>
                            <label>Работа матери (Mjob):</label>
                            <select id="Mjob" name="Mjob">
                                <option value="teacher">Учитель</option>
                                <option value="health">Здравоохранение</option>
                                <option value="services" selected>Сфера услуг</option>
                                <option value="at_home">Домохозяйка</option>
                                <option value="other">Другое</option>
                            </select>
                        </div>
                        <div>
                            <label>Работа отца (Fjob):</label>
                            <select id="Fjob" name="Fjob">
                                <option value="teacher">Учитель</option>
                                <option value="health">Здравоохранение</option>
                                <option value="services">Сфера услуг</option>
                                <option value="at_home">Дома</option>
                                <option value="other" selected>Другое</option>
                            </select>
                        </div>
                        <div>
                            <label>Опекун (guardian):</label>
                            <select id="guardian" name="guardian">
                                <option value="mother" selected>Мать</option>
                                <option value="father">Отец</option>
                                <option value="other">Другой</option>
                            </select>
                        </div>
                        <div>
                            <label>Детский сад (nursery):</label>
                            <select id="nursery" name="nursery">
                                <option value="yes" selected>Посещал</option>
                                <option value="no">Не посещал</option>
                            </select>
                        </div>
                    </div>
                </div>

                <!-- 4. ОБРАЗ ЖИЗНИ И ЗДОРОВЬЕ -->
                <div class="section-box">
                    <div class="section-title">Образ жизни и здоровье</div>
                    <div class="grid">
                        <div>
                            <label>Отношения в семье (1-5):</label>
                            <select id="famrel" name="famrel">
                                <option value="1">1 - Очень плохие</option>
                                <option value="2">2 - Плохие</option>
                                <option value="3">3 - Средние</option>
                                <option value="4" selected>4 - Хорошие</option>
                                <option value="5">5 - Отличные</option>
                            </select>
                        </div>
                        <div>
                            <label>Свободное время (1-5):</label>
                            <select id="freetime" name="freetime">
                                <option value="1">1 - Очень мало</option>
                                <option value="2">2 - Мало</option>
                                <option value="3" selected>3 - Умеренно</option>
                                <option value="4">4 - Много</option>
                                <option value="5">5 - Очень много</option>
                            </select>
                        </div>
                        <div>
                            <label>Встречи с друзьями (1-5):</label>
                            <select id="goout" name="goout">
                                <option value="1">1 - Очень редко</option>
                                <option value="2">2 - Редко</option>
                                <option value="3" selected>3 - Умеренно</option>
                                <option value="4">4 - Часто</option>
                                <option value="5">5 - Очень часто</option>
                            </select>
                        </div>
                        <div>
                            <label>Алкоголь в будни (Dalc):</label>
                            <select id="Dalc" name="Dalc">
                                <option value="1" selected>1 - Очень низкое</option>
                                <option value="2">2 - Низкое</option>
                                <option value="3">3 - Среднее</option>
                                <option value="4">4 - Высокое</option>
                                <option value="5">5 - Очень высокое</option>
                            </select>
                        </div>
                        <div>
                            <label>Алкоголь в выходные (Walc):</label>
                            <select id="Walc" name="Walc">
                                <option value="1" selected>1 - Очень низкое</option>
                                <option value="2">2 - Низкое</option>
                                <option value="3">3 - Среднее</option>
                                <option value="4">4 - Высокое</option>
                                <option value="5">5 - Очень высокое</option>
                            </select>
                        </div>
                        <div>
                            <label>Самооценка здоровья (health):</label>
                            <select id="health" name="health">
                                <option value="1">1 - Очень плохое</option>
                                <option value="2">2 - Плохое</option>
                                <option value="3" selected>3 - Нормальное</option>
                                <option value="4">4 - Хорошее</option>
                                <option value="5">5 - Отличное</option>
                            </select>
                        </div>
                        <div>
                            <label>Внеклассные занятия (activities):</label>
                            <select id="activities" name="activities">
                                <option value="no" selected>Нет</option>
                                <option value="yes">Да</option>
                            </select>
                        </div>
                        <div>
                            <label>Интернет дома (internet):</label>
                            <select id="internet" name="internet">
                                <option value="yes" selected>Есть</option>
                                <option value="no">Нет</option>
                            </select>
                        </div>
                        <div>
                            <label>Романтические отношения:</label>
                            <select id="romantic" name="romantic">
                                <option value="no" selected>Нет</option>
                                <option value="yes">Есть</option>
                            </select>
                        </div>
                    </div>
                </div>

                <button type="button" class="btn-submit" onclick="submitPrediction()">Рассчитать риск неуспеваемости</button>
            </form>

            <div id="resultCard" class="result-card">
                <div id="riskBadge" style="font-weight: 700; font-size: 14px; text-transform: uppercase;"></div>
                <div id="probValue" class="prob-value"></div>
                <div id="recommendationText" style="font-size: 15px; font-weight: 500;"></div>
            </div>
        </div>

        <script>
            function setPreset(type) {
                if (type === 'good') {
                    document.getElementById('school').value = 'GP';
                    document.getElementById('failures').value = '0';
                    document.getElementById('absences').value = '2';
                    document.getElementById('studytime').value = '3';
                    document.getElementById('higher').value = 'yes';
                    document.getElementById('Dalc').value = '1';
                    document.getElementById('Walc').value = '1';
                    document.getElementById('Medu').value = '4';
                    document.getElementById('Fedu').value = '4';
                } else {
                    document.getElementById('school').value = 'MS';
                    document.getElementById('failures').value = '2';
                    document.getElementById('absences').value = '16';
                    document.getElementById('studytime').value = '1';
                    document.getElementById('higher').value = 'no';
                    document.getElementById('Dalc').value = '3';
                    document.getElementById('Walc').value = '4';
                    document.getElementById('Medu').value = '1';
                    document.getElementById('Fedu').value = '1';
                }
                submitPrediction();
            }

            async function submitPrediction() {
                const form = document.getElementById('studentForm');
                const formData = new FormData(form);
                const payload = {};

                for (let [key, value] of formData.entries()) {
                    if (['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'].includes(key)) {
                        payload[key] = parseInt(value) || 0;
                    } else {
                        payload[key] = value;
                    }
                }

                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await res.json();
                const card = document.getElementById('resultCard');
                card.style.display = 'block';

                const pct = (data.failure_probability * 100).toFixed(1);
                document.getElementById('probValue').innerText = pct + '%';
                document.getElementById('recommendationText').innerText = 'Рекомендация: ' + data.recommendation;

                if (data.risk_group) {
                    card.className = 'result-card result-danger';
                    document.getElementById('riskBadge').innerText = 'Внимание: Высокий риск академической задолженности';
                } else {
                    card.className = 'result-card result-safe';
                    document.getElementById('riskBadge').innerText = 'Низкий риск: Успеваемость в норме';
                }

                card.scrollIntoView({ behavior: 'smooth' });
            }
        </script>
    </body>
    </html>
    """