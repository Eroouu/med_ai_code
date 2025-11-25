from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from run_enchance import MedicalInterviewBot
from fastapi.responses import StreamingResponse
import traceback

app = FastAPI()

# ==================== CORS ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ИНИЦИАЛИЗАЦИЯ БОТА ====================
print("\n🚀 Инициализация сервера...")
print("⏳ Загрузка медицинского бота...\n")

try:
    bot = MedicalInterviewBot(rebuild_db=False)
    print("✅ Бот готов!\n")
except Exception as e:
    print(f"❌ Ошибка: {e}\n")
    raise

# ==================== МОДЕЛИ ДАННЫХ ====================

class DemographicsRequest(BaseModel):
    """ЭТАП 1: Данные из таблицы демографии"""
    name: str
    age: int
    weight: float
    height: int

class GeneralQuestionAnswerRequest(BaseModel):
    """ЭТАП 2: Ответ на общий вопрос"""
    session_id: str
    question_key: str
    answer: str

class SymptomMessageRequest(BaseModel):
    """ЭТАП 3: Сообщение в беседе о симптомах"""
    session_id: str
    message: str

# ==================== ENDPOINTS ====================

@app.get("/")
async def get_index():
    """Главная страница"""
    try:
        return FileResponse("templates/index.html", media_type="text/html; charset=utf-8")
    except FileNotFoundError:
        return {"error": "Template not found"}

@app.post("/api/session/start")
async def start_session():
    """🆕 Создает новую сессию для пациента"""
    try:
        session_id = bot.create_session()
        print(f"\n✅ Новая сессия: {session_id}")
        
        return {
            "status": "ok",
            "session_id": session_id,
            "stage": "demographics",
            "message": "Сессия создана. Начните с заполнения таблицы демографии."
        }
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

# ==================== ЭТАП 1: ДЕМОГРАФИЯ ====================

@app.post("/api/session/demographics")
async def save_demographics(session_id: str, request: DemographicsRequest):
    """📋 ЭТАП 1: Сохранение демографических данных из таблицы"""
    try:
        print(f"\n📋 ЭТАП 1: Сохранение демографии")
        print(f"   Пациент: {request.name}, {request.age} лет, {request.weight} кг, {request.height} см")
        
        demographics = {
            "name": request.name,
            "age": request.age,
            "weight": request.weight,
            "height": request.height
        }
        
        success = bot.save_demographics(session_id, demographics)
        
        if success:
            questions = bot.get_general_questions()
            first_question = questions[0]
            
            print(f"✅ Демография сохранена")
            print(f"➡️ Переход на ЭТАП 2: {first_question['question']}")
            
            return {
                "status": "ok",
                "stage": "general_questions",
                "message": "Демография сохранена!",
                "next_question": first_question
            }
        else:
            return {"status": "error", "message": "Не удалось сохранить демографию"}
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

# ==================== ЭТАП 2: ОБЩИЕ ВОПРОСЫ ====================

@app.get("/api/session/general_questions")
async def get_general_questions():
    """💬 Возвращает список всех 5 общих вопросов для ЭТАПА 2"""
    try:
        questions = bot.get_general_questions()
        return {
            "status": "ok",
            "questions": questions,
            "total": len(questions)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/session/general_questions/answer")
async def save_general_question_answer(request: GeneralQuestionAnswerRequest):
    """✍️ ЭТАП 2: Сохранение ответа на общий вопрос с нормализацией"""
    try:
        print(f"\n💬 ЭТАП 2: Ответ на вопрос '{request.question_key}'")
        print(f"   Пациент ответил: {request.answer}")
        
        # Нормализуем ответ через ЛЛМ
        normalized_answer = await bot.extract_demographics_hybrid(
            field_name=request.question_key,
            answer=request.answer
        )
        print(f"   Нормализовано: {normalized_answer}")
        
        # Сохраняем ответ
        success = await bot.save_general_question_answer(
            session_id=request.session_id,
            question_key=request.question_key,
            answer=normalized_answer
        )
        
        if not success:
            return {"status": "error", "message": "Не удалось сохранить ответ"}
        
        # Проверяем, все ли вопросы ответили
        session = bot.load_session(request.session_id)
        all_answered = all(v is not None for v in session["general_questions"].values())
        
        response = {
            "status": "ok",
            "original": request.answer,
            "normalized": normalized_answer,
            "saved": True,
            "all_answered": all_answered,
            "stage": session["stage"]
        }
        
        if all_answered:
            print(f"✅ Все общие вопросы ответили!")
            print(f"➡️ Переход на ЭТАП 3: Сбор информации о жалобе")
            response["message"] = "Все общие вопросы ответили! Переходим к следующему этапу."
        else:
            # Возвращаем следующий вопрос
            questions = bot.get_general_questions()
            answered_keys = [k for k, v in session["general_questions"].items() if v is not None]
            for q in questions:
                if q["key"] not in answered_keys:
                    response["next_question"] = q
                    response["message"] = f"Ответ сохранен. Следующий вопрос готов."
                    break
        
        return response
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

# ==================== ЭТАП 3: ЖАЛОБА И СИМПТОМЫ ====================

@app.get("/api/session/symptoms/initial_question")
async def get_initial_symptoms_question(session_id: str):
    """🩺 ЭТАП 3: Получить первый вопрос о жалобе (фиксированный)"""
    try:
        print(f"\n🩺 ЭТАП 3: Первый вопрос о жалобе")
        
        initial_question = await bot.get_initial_symptoms_question()
        
        return {
            "status": "ok",
            "question": initial_question,
            "is_first": True,
            "message": "Начинаем сбор информации о жалобе"
        }
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/session/symptoms/message")
async def handle_symptoms_message(request: SymptomMessageRequest):
    """💬 ЭТАП 3: Обработка сообщения о симптомах с динамическими вопросами"""
    try:
        session = bot.load_session(request.session_id)
        if not session:
            return {"status": "error", "message": "Сессия не найдена"}
        
        # Если это первое сообщение пациента - сохраняем как основную жалобу
        if session["symptoms"]["chief_complaint"] is None:
            print(f"\n🩺 ЭТАП 3: Основная жалоба получена")
            print(f"   Пациент: {request.message}")
            
            bot.save_chief_complaint(request.session_id, request.message)
            bot.save_conversation_message(request.session_id, "patient", request.message)
            
            # Генерируем первый уточняющий вопрос
            bot_question = await bot.generate_symptoms_question(request.session_id)
            bot.save_conversation_message(request.session_id, "bot", bot_question)
            
            print(f"   Бот спрашивает: {bot_question}")
            
            return {
                "status": "ok",
                "bot_reply": bot_question,
                "should_continue": True,
                "question_count": 1,
                "message": "Жалоба получена. Бот задает уточняющий вопрос."
            }
        
        # Сохраняем ответ пациента
        print(f"\n🩺 ЭТАП 3: Ответ пациента")
        print(f"   {request.message}")
        
        bot.save_conversation_message(request.session_id, "patient", request.message)
        
        # Проверяем, достаточно ли информации
        should_stop = await bot.should_stop_conversation(request.session_id)
        
        if should_stop:
            print(f"\n✅ ЭТАП 3 завершен: достаточно информации собрано")
            print(f"📊 Генерирую отчет для врача...")
            
            report = bot.generate_text_report(request.session_id)
            session = bot.load_session(request.session_id)
            
            return {
                "status": "ok",
                "bot_reply": "✅ Спасибо за информацию! Все необходимые данные собраны.",
                "should_continue": False,
                "question_count": session["symptoms"]["question_count"],
                "stage": "completed",
                "report": report,
                "session_data": bot.get_session_report(request.session_id),
                "message": "Беседа завершена. Отчет готов для врача."
            }
        
        # Генерируем следующий вопрос
        bot_question = await bot.generate_symptoms_question(request.session_id)
        bot.save_conversation_message(request.session_id, "bot", bot_question)
        
        session = bot.load_session(request.session_id)
        print(f"   Бот спрашивает: {bot_question}")
        print(f"   Всего вопросов: {session['symptoms']['question_count']}")
        
        return {
            "status": "ok",
            "bot_reply": bot_question,
            "should_continue": True,
            "question_count": session["symptoms"]["question_count"],
            "message": "Вопрос сохранен. Ждем следующий ответ пациента."
        }
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

# ==================== ПОЛУЧЕНИЕ ОТЧЕТОВ ====================

@app.get("/api/session/report")
async def get_session_report(session_id: str):
    """📊 Получить полный отчет сессии в формате JSON"""
    try:
        report = bot.get_session_report(session_id)
        if report:
            return {"status": "ok", "report": report}
        else:
            return {"status": "error", "message": "Сессия не найдена"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/session/report/text")
async def get_session_report_text(session_id: str):
    """📄 Получить текстовый отчет для врача"""
    try:
        report = bot.generate_text_report(session_id)
        if report:
            return {"status": "ok", "text_report": report}
        else:
            return {"status": "error", "message": "Сессия не найдена"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==================== УПРАВЛЕНИЕ СЕССИЯМИ ====================

@app.delete("/api/session")
async def delete_session(session_id: str):
    """🗑️ Удалить сессию"""
    try:
        bot.delete_session(session_id)
        return {"status": "ok", "message": f"Сессия {session_id} удалена"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/health")
async def health():
    """❤️ Проверка здоровья сервера"""
    return {
        "status": "healthy",
        "service": "Medical Interview Bot",
        "version": "2.0",
        "stages": [
            "1. demographics",
            "2. general_questions",
            "3. symptoms"
        ]
    }

@app.get("/api/session/report/download")
async def download_report(session_id: str):
    """
    📥 СКАЧИВАНИЕ ОТЧЕТА
    Возвращает медицинский отчет в текстовом формате
    Файл скачивается как: medical_report_{session_id}.txt
    """
    try:
        # Генерируем текстовый отчет
        report = bot.generate_text_report(session_id)
        
        if not report:
            return {"status": "error", "message": "Сессия не найдена"}
        
        # Возвращаем файл для скачивания
        from io import BytesIO
        from datetime import datetime
        
        # Добавляем метаданные в начало файла
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_report = f"═══════════════════════════════════════════════════════════\n"
        full_report += f"МЕДИЦИНСКИЙ ОТЧЕТ\n"
        full_report += f"Сгенерирован: {timestamp}\n"
        full_report += f"ID сессии: {session_id}\n"
        full_report += f"═══════════════════════════════════════════════════════════\n\n"
        full_report += report
        
        # Конвертируем в bytes
        report_bytes = full_report.encode('utf-8')
              
        # Возвращаем файл
        filename = f"medical_report_{session_id[:8]}.txt"
        
        print(f"📥 Скачивание отчета: {filename}")
        
        return StreamingResponse(
            iter([report_bytes]),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        print(f"❌ Ошибка скачивания: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
    
# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🎯 МЕДИЦИНСКИЙ АССИСТЕНТ - СЕРВЕР ЗАПУЩЕН")
    print("="*70)
    print("\n📋 ТРИ ЭТАПА:")
    print("  1️⃣  ЭТАП 1: Таблица демографии (имя, возраст, вес, рост)")
    print("  2️⃣  ЭТАП 2: Общие вопросы (5 вопросов с нормализацией)")
    print("  3️⃣  ЭТАП 3: Жалоба пациента (динамическая беседа с ботом)")
    print("\n🌐 Адрес: http://localhost:8000")
    print("📚 API: http://localhost:8000/docs")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="localhost", port=8000)
