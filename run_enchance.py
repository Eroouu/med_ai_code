from pathlib import Path
import json
import tempfile
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import asyncio
import uuid


class MedicalInterviewBot:
    """
    Медицинский интервью-бот с 3 этапами:
    
    1️⃣  ЭТАП 1: ТАБЛИЦА ДЕМОГРАФИИ
        - Имя, Возраст, Вес, Рост
        - Данные вносятся в таблицу и сохраняются в JSON сессию
    
    2️⃣  ЭТАП 2: ОБЩИЕ ВОПРОСЫ
        - 5 вопросов через чат: medications, allergies, surgeries, chronic_diseases, lifestyle
        - Пациент пишет свой ответ
        - ЛЛМ нормализует (extract_demographics_hybrid) в официальный вид
        - Сохраняются нормализованные ответы
    
    3️⃣  ЭТАП 3: ЖАЛОБА И СИМПТОМЫ
        - Первый вопрос: "Добрый день, скажите пожалуйста что вас беспокоит?"
        - Остальные вопросы генерируются ЛЛМом для уточнения
        - Беседа продолжается пока не соберется достаточно информации
        - Результат - быстрый отчет для врача
    """

    def __init__(self, rebuild_db=False):
        """Инициализация бота"""
        
        # ==================== ПУТИ И КОНФИГ ====================
        self.script_dir = Path(__file__).parent
        self.data_dir = self.script_dir / "enhanced_dataset"
        self.sessions_dir = self.script_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        
        temp_base = Path(tempfile.gettempdir())
        self.db_dir = temp_base / "medical_bot_db"

        # ==================== LLM И БД ====================
        self.llm = ChatOllama(model="llama3.1", temperature=0.7)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = None

        # Инициализируем БД
        self._load_or_create_knowledge_base(rebuild_db)

    # ==================== БД: FAISS ====================

    def _load_or_create_knowledge_base(self, rebuild_db=False):
        """Загружает или создаёт FAISS БД"""
        
        if rebuild_db and self.db_dir.exists():
            import shutil
            shutil.rmtree(self.db_dir)
            print("✅ БД удалена для перестройки")

        if self.db_dir.exists() and (self.db_dir / "index.faiss").exists():
            print(f"📂 Загружаю FAISS БД из {self.db_dir}...")
            try:
                self.vectorstore = FAISS.load_local(
                    str(self.db_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ БД загружена успешно!")
                return
            except Exception as e:
                print(f"⚠️ Ошибка загрузки БД: {e}")

        # Создаём новую БД
        self._create_new_database(self.embeddings)

    def _create_new_database(self, embeddings):
        """Создаёт новую FAISS БД из JSON файлов"""
        
        print("🔨 Создаю новую БД...")
        documents = []

        # Загружаем JSON файлы
        json_files = list(self.data_dir.glob("*.json"))
        if not json_files:
            print(f"❌ JSON файлы не найдены в {self.data_dir}!")
            return

        total = len(json_files)
        print(f"📄 Найдено {total} файлов")

        # Парсим и индексируем документы
        for i, json_file in enumerate(json_files, 1):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                title = data.get("title", "").strip()
                sections = data.get("sections", {})
                full_text = f"{title}\n"

                # Собираем текст
                for section_name, section_text in sections.items():
                    if not section_text or not str(section_text).strip():
                        continue
                    readable_name = section_name.replace("_", " ").title()
                    full_text += f"\n{readable_name}\n{section_text}\n"

                if len(full_text) < 100:
                    continue

                # Метаданные
                meta = data.get("metadata", {})
                doc_metadata = {
                    "title": title,
                    "disease": title,
                    "file": json_file.name,
                    "categories": meta.get("categories", ""),
                    "symptoms": meta.get("symptoms", ""),
                    "complexity": meta.get("complexity", ""),
                }

                documents.append(Document(
                    page_content=full_text,
                    metadata=doc_metadata
                ))

                if i % 50 == 0 or i == total:
                    print(f" ✓ Обработано {i}/{total}")

            except Exception as e:
                print(f"⚠️ Ошибка при обработке {json_file.name}: {e}")

        print(f"✅ Документов загружено: {len(documents)}")

        # Разбиваем на chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)
        total_splits = len(splits)
        print(f"✂️ Разбиено на {total_splits} chunks")

        # Создаём FAISS
        print("🔍 Индексирую документы...")
        vectorstore = None
        batch_size = 100

        for i in range(0, total_splits, batch_size):
            batch = splits[i:i+batch_size]
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, self.embeddings)
            else:
                vectorstore.add_documents(batch)

            progress = min(i + batch_size, total_splits)
            percent = (progress / total_splits) * 100
            print(f" ✓ {progress}/{total_splits} ({percent:.1f}%)")

        self.vectorstore = vectorstore
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.db_dir))
        print(f"💾 БД сохранена в {self.db_dir}")

    def search_context(self, query: str, k: int = 3) -> str:
        """Ищет контекст в БД"""
        
        if not self.vectorstore or not query:
            return ""

        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return "\n---\n".join(doc.page_content[:700] for doc in docs)
        except Exception as e:
            print(f"⚠️ Ошибка поиска: {e}")
            return ""

    # ==================== УПРАВЛЕНИЕ СЕССИЯМИ ====================

    def create_session(self) -> str:
        """
        Создает новую сессию для пациента
        Возвращает session_id
        """
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "stage": "demographics",
            "demographics": {
                "name": None,
                "age": None,
                "weight": None,
                "height": None
            },
            "general_questions": {
                "medications": None,
                "allergies": None,
                "surgeries": None,
                "chronic_diseases": None,
                "lifestyle": None
            },
            "symptoms": {
                "chief_complaint": None,
                "conversation": [],
                "question_count": 0
            }
        }
        self.save_session(session_id, session_data)
        print(f"✅ Новая сессия создана: {session_id}")
        return session_id

    def save_session(self, session_id: str, data: dict):
        """
        Сохраняет сессию в JSON файл
        Вызывается после каждого этапа для персистентности
        """
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            data["updated_at"] = datetime.now().isoformat()
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"💾 Сессия сохранена: {session_id}")
        except Exception as e:
            print(f"❌ Ошибка сохранения сессии: {e}")

    def load_session(self, session_id: str) -> dict:
        """Загружает сессию из JSON файла"""
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                with open(session_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                print(f"❌ Сессия не найдена: {session_id}")
                return None
        except Exception as e:
            print(f"❌ Ошибка загрузки сессии: {e}")
            return None

    def delete_session(self, session_id: str):
        """Удаляет сессию"""
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
                print(f"🗑️ Сессия удалена: {session_id}")
        except Exception as e:
            print(f"❌ Ошибка удаления сессии: {e}")

    # ==================== ЭТАП 1: ДЕМОГРАФИЯ ====================

    def save_demographics(self, session_id: str, demographics: dict) -> bool:
        """
        Сохраняет демографические данные (таблица)
        demographics = {name, age, weight, height}
        Переводит на ЭТАП 2 (общие вопросы)
        """
        session = self.load_session(session_id)
        if not session:
            return False
        
        session["demographics"] = demographics
        session["stage"] = "general_questions"
        self.save_session(session_id, session)
        print(f"✅ ЭТАП 1 завершен: демография сохранена")
        return True

    # ==================== ЭТАП 2: ОБЩИЕ ВОПРОСЫ ====================

    def get_general_questions(self) -> list:
        """
        Возвращает список из 5 вопросов для ЭТАПА 2
        Один вопрос за раз - пациент отвечает, бот нормализует, переходит к следующему
        """
        return [
            {
                "key": "medications",
                "question": "💊 Какие лекарства вы принимаете в настоящее время?"
            },
            {
                "key": "allergies",
                "question": "🚫 У вас есть аллергии на какие-либо вещества или лекарства?"
            },
            {
                "key": "surgeries",
                "question": "🏥 Были ли у вас когда-либо операции?"
            },
            {
                "key": "chronic_diseases",
                "question": "⚕️ Есть ли у вас хронические заболевания или постоянные проблемы со здоровьем?"
            },
            {
                "key": "lifestyle",
                "question": "🏃 Как бы вы охарактеризовали свой образ жизни? (активный/умеренный/малоподвижный)"
            }
        ]

    async def save_general_question_answer(self, session_id: str, question_key: str, answer: str):
        """
        Сохраняет ответ на общий вопрос (ЭТАП 2)
        1. Нормализует ответ через extract_demographics_hybrid
        2. Сохраняет нормализованный ответ
        3. Если все 5 вопросов ответили - переводит на ЭТАП 3
        """
        session = self.load_session(session_id)
        if not session:
            return False
        
        # Нормализуем ответ
        normalized_answer = await self.extract_demographics_hybrid(question_key, answer)
        session["general_questions"][question_key] = normalized_answer
        
        # Проверяем, все ли вопросы ответили
        all_answered = all(v is not None for v in session["general_questions"].values())
        
        if all_answered:
            session["stage"] = "symptoms"
            print(f"✅ ЭТАП 2 завершен: все общие вопросы ответили")
            print(f"➡️ Переход на ЭТАП 3: сбор информации о жалобе")
        
        self.save_session(session_id, session)
        return True

    # ==================== ЭТАП 2: НОРМАЛИЗАЦИЯ ОТВЕТОВ ====================

    async def extract_demographics_hybrid(self, field_name: str, answer: str) -> str:
        """
        КЛЮЧЕВАЯ ФУНКЦИЯ: Нормализует ответы пациента в официальный медицинский вид для отчета
        
        Работает для всех 5 полей общих вопросов:
        - medications (лекарства)
        - allergies (аллергии)
        - surgeries (операции)
        - chronic_diseases (хронические болезни)
        - lifestyle (образ жизни)
        
        Примеры:
        --------
        Вход: field_name="medications", answer="ну вот пью таблетки какие-то для щитовидки"
        Выход: "Гормональные препараты для лечения щитовидной железы (Левотироксин)"
        
        Вход: field_name="allergies", answer="у меня реакция на антибиотики пенициллинового ряда"
        Выход: "Аллергия на антибиотики пенициллинового ряда"
        
        Вход: field_name="lifestyle", answer="сижу дома, не активный совсем"
        Выход: "Малоподвижный образ жизни"
        """
        
        # ===== ПРОВЕРКА ПУСТОГО ОТВЕТА =====
        if not answer or not answer.strip():
            return "Не указано"

        # ===== ПРОВЕРКА "НЕТ" В РАЗНЫХ ВАРИАНТАХ =====
        if answer.lower() in ["нет", "no", "none", "-", "не", "ничего", "не указано"]:
            return "Не отмечается"

        # ===== ЛЛМ НОРМАЛИЗАЦИЯ =====
        prompts = {
            "medications": """Ты медицинский помощник. Нормализуй ответ пациента о принимаемых лекарствах в официальный медицинский вид.

Ответ пациента: "{answer}"

Правила:
- Напиши название активного вещества или класса препаратов
- Добавь скобки с примером: (Парацетамол, Ибупрофен)
- Если указана причина - добавь "для лечения..."
- Будь лаконичен (макс 1-2 строки)
- Если неясно - напиши "Нет данных"

Верни ТОЛЬКО нормализованный ответ без кавычек и пояснений.""",

            "allergies": """Ты медицинский помощник. Нормализуй ответ пациента об аллергиях в официальный медицинский вид.

Ответ пациента: "{answer}"

Правила:
- Напиши тип аллергии (лекарственная, пищевая, на вещество и т.д.)
- Укажи конкретное вещество если упомянуто
- Напиши реакцию если известна: (сыпь, отек, анафилаксия и т.д.)
- Если ничего не уточнено - напиши "Не отмечается"
- Будь лаконичен

Верни ТОЛЬКО нормализованный ответ без кавычек и пояснений.""",

            "surgeries": """Ты медицинский помощник. Нормализуй ответ пациента об операциях в официальный медицинский вид.

Ответ пациента: "{answer}"

Правила:
- Напиши название операции в медицинском стиле
- Если известен год - добавь в скобки: (2015 год)
- Если не уточнено - напиши "Не указано"
- Если нет операций - напиши "Не проводились"
- Будь лаконичен

Верни ТОЛЬКО нормализованный ответ без кавычек и пояснений.""",

            "chronic_diseases": """Ты медицинский помощник. Нормализуй ответ пациента о хронических заболеваниях в официальный медицинский вид.

Ответ пациента: "{answer}"

Правила:
- Напиши диагноз в медицинском стиле (латинское название если известно)
- Укажи статус лечения если известен: (в лечении, контролируется и т.д.)
- Если не уточнено - напиши "Не указано"
- Если нет болезней - напиши "Не отмечаются"
- Будь лаконичен

Верни ТОЛЬКО нормализованный ответ без кавычек и пояснений.""",

            "lifestyle": """Ты медицинский помощник. Нормализуй ответ пациента об образе жизни в официальный медицинский вид.

Ответ пациента: "{answer}"

Правила:
- Классифицируй как: Активный / Умеренный / Малоподвижный
- Добавь детали если есть: (спорт, работа за ПК и т.д.)
- Упомяни привычки если они явно вредные: (курение, алкоголь)
- Если неясно - напиши "Не уточнено"
- Будь лаконичен

Верни ТОЛЬКО нормализованный ответ без кавычек и пояснений."""
        }

        prompt_template = prompts.get(field_name, 
            """Ты медицинский помощник. Нормализуй ответ пациента в официальный медицинский вид.

Ответ пациента: "{answer}"

Верни ТОЛЬКО нормализованный ответ без кавычек и пояснений.""")

        prompt = prompt_template.format(answer=answer)

        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)  # ← ГЛАВНОЕ ИЗМЕНЕНИЕ
            normalized = response.content.strip()
            if not normalized:
                return "Не указано"
            print(f"✅ Нормализовано ({field_name}): {answer[:40]}... → {normalized[:50]}...")
            return normalized
        except Exception as e:
            print(f"❌ ОШИБКА extract_demographics_hybrid ({field_name}): {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return answer if answer.strip() else "Не указано"

    # ==================== ЭТАП 3: ЖАЛОБА И СИМПТОМЫ ====================

    async def get_initial_symptoms_question(self) -> str:
        """
        Первый вопрос на ЭТАПЕ 3 - фиксированный
        "Добрый день, скажите пожалуйста что вас беспокоит?"
        """
        return "Добрый день! Скажите, пожалуйста, что вас беспокоит?"

    def save_chief_complaint(self, session_id: str, complaint: str):
        """
        Сохраняет основную жалобу пациента
        После первого сообщения пациента на ЭТАПЕ 3
        """
        session = self.load_session(session_id)
        if not session:
            return False
        
        session["symptoms"]["chief_complaint"] = complaint
        self.save_session(session_id, session)
        print(f"✅ Основная жалоба сохранена: {complaint[:50]}...")
        return True

    async def generate_symptoms_question(self, session_id: str) -> str:
        """
        Генерирует следующий вопрос для уточнения жалобы (ЭТАП 3)
        Анализирует:
        1. Основную жалобу
        2. История переписки
        3. Уже собранная информация
        
        Возвращает релевантный уточняющий вопрос
        """
        session = self.load_session(session_id)
        if not session:
            return "Опишите ваши симптомы подробнее."
        
        chief_complaint = session["symptoms"]["chief_complaint"]
        conversation = session["symptoms"]["conversation"]
        question_count = session["symptoms"]["question_count"]
        
        # Формируем контекст для ЛЛМ
        history = "\n".join([f"- {msg['role']}: {msg['content']}" for msg in conversation[-6:]])
        
        prompt = f"""Ты опытный врач. На основе жалобы пациента и истории беседы, 
сгенерируй следующий уточняющий вопрос для правильной диагностики. Твоя задача собрать как можно больше общей информации о болезни пациента,
а не поставить диагноз!

ЖАЛОБА: "{chief_complaint}"

ИСТОРИЯ БЕСЕДЫ (последние ответы):
{history}

ПРАВИЛА:
- Задай ОДИН конкретный медицинский вопрос
- Избегай повторений - не спрашивай то же, что уже спрашивал
- Вопрос должен помочь уточнить: характер боли, длительность, интенсивность, сопутствующие симптомы
- Будь конкретен и профессионален
- Максимум одно предложение

Верни ТОЛЬКО вопрос без пояснений."""

        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)  # ← ГЛАВНОЕ ИЗМЕНЕНИЕ
            question = response.content.strip()
            if not question or len(question) < 5:
                return "Расскажите подробнее о ваших ощущениях."
            print(f"✅ Вопрос сгенерирован: {question[:60]}...")
            return question
        except Exception as e:
            print(f"❌ ОШИБКА generate_symptoms_question: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return "Расскажите подробнее о ваших ощущениях."

    def save_conversation_message(self, session_id: str, role: str, content: str):
        """
        Сохраняет сообщение в беседу (ЭТАП 3)
        role: "patient" или "bot"
        """
        session = self.load_session(session_id)
        if not session:
            return False
        
        session["symptoms"]["conversation"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        if role == "bot":
            session["symptoms"]["question_count"] += 1
        
        self.save_session(session_id, session)
        return True

    async def should_stop_conversation(self, session_id: str) -> bool:
        """
        Проверяет, достаточно ли информации для завершения ЭТАПА 3
        Возвращает True если пора завершить, False если продолжить
        
        Критерии:
        - Минимум 4 вопроса-ответа (8 сообщений)
        - Максимум 10 вопросов-ответов (20 сообщений)
        - ЛЛМ анализ: достаточно ли информации?
        """
        session = self.load_session(session_id)
        if not session:
            return False
        
        conversation = session["symptoms"]["conversation"]
        question_count = session["symptoms"]["question_count"]
        
        # Жесткие пределы
        if question_count < 3:
            return False
        if question_count >= 10:
            return True
        
        # ЛЛМ проверка
        chief_complaint = session["symptoms"]["chief_complaint"]
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation[-8:]])
        
        prompt = f"""Ты врач. Проанализируй беседу с пациентом.

ЖАЛОБА: "{chief_complaint}"

БЕСЕДА:
{history}

Вопрос: Собрано ли достаточно информации для быстрого отчета врачу?

КРИТЕРИИ достаточности:
- Понимание характера боли/жалобы
- Длительность симптомов
- Интенсивность
- Сопутствующие симптомы
- Провоцирующие факторы

Ответь ТОЛЬКО "ДА" или "НЕТ"."""

        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)  # ← ГЛАВНОЕ ИЗМЕНЕНИЕ
            answer = response.content.strip().upper()
            result = answer == "ДА"
            print(f"{'✅' if result else '⏳'} Проверка завершения: {answer}")
            return result
        except Exception as e:
            print(f"❌ ОШИБКА should_stop_conversation: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return question_count >= 6

    # ==================== ИТОГОВЫЕ ОТЧЕТЫ ====================

    def get_session_report(self, session_id: str) -> dict:
        """Возвращает полный отчет сессии для врача"""
        session = self.load_session(session_id)
        if not session:
            return None
        
        return {
            "session_id": session_id,
            "created_at": session.get("created_at"),
            "demographics": session.get("demographics"),
            "general_questions": session.get("general_questions"),
            "symptoms": session.get("symptoms"),
            "stage": session.get("stage")
        }

    def generate_text_report(self, session_id: str) -> str:
        """Генерирует красивый текстовый отчет для врача"""
        session = self.load_session(session_id)
        if not session:
            return "❌ Сессия не найдена"
        
        demo = session.get("demographics", {})
        gen_q = session.get("general_questions", {})
        symp = session.get("symptoms", {})
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
╔═══════════════════════════════════════════════════════════╗
║           МЕДИЦИНСКИЙ АНАМНЕЗ ПАЦИЕНТА                    ║
╚═══════════════════════════════════════════════════════════╝

📋 ДАТА И ВРЕМЯ: {now}
🔑 ID СЕССИИ: {session_id}

👤 ЛИЧНЫЕ ДАННЫЕ:
   Имя: {demo.get('name', '—')}
   Возраст: {demo.get('age', '—')} лет
   Вес: {demo.get('weight', '—')} кг
   Рост: {demo.get('height', '—')} см

🗣️ ОСНОВНАЯ ЖАЛОБА:
   {symp.get('chief_complaint', '—')}

💊 ЛЕКАРСТВА:
   {gen_q.get('medications', '—')}

🚫 АЛЛЕРГИИ:
   {gen_q.get('allergies', '—')}

🏥 ОПЕРАЦИИ:
   {gen_q.get('surgeries', '—')}

⚕️ ХРОНИЧЕСКИЕ ЗАБОЛЕВАНИЯ:
   {gen_q.get('chronic_diseases', '—')}

🏃 ОБРАЗ ЖИЗНИ:
   {gen_q.get('lifestyle', '—')}

📝 ИСТОРИЯ БЕСЕДЫ ДЛЯ УТОЧНЕНИЯ:
"""
        for msg in symp.get("conversation", []):
            role_emoji = "🤖" if msg["role"] == "bot" else "👤"
            report += f"\n   {role_emoji} {msg['role'].upper()}: {msg['content']}"
        
        report += "\n\n═══════════════════════════════════════════════════════════\n"
        return report


# ==================== ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ====================

if __name__ == "__main__":
    print("🏥 Медицинский ассистент загружен")
    print("Используйте этот класс в FastAPI приложении")
