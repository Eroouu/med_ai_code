from pathlib import Path
import json
import tempfile
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


class MedicalInterviewBot:
    def __init__(self, rebuild_db: bool = False):
        self.script_dir = Path(__file__).parent
        # Новый формат датасета
        self.data_dir = self.script_dir / "enhanced_dataset"

        # Временная папка для FAISS‑индекса
        temp_base = Path(tempfile.gettempdir())
        self.db_dir = temp_base / "medical_bot_db"

        self.conversation_history = []
        self.collected_info = {
            "chief_complaint": "",
            "symptoms": [],
            "duration": "",
            "additional_info": []
        }

        print("=" * 70)
        print("🏥 МЕДИЦИНСКИЙ ИНТЕРВЬЮЕР v2.6 (enhanced_dataset)")
        print("=" * 70)
        print(f"\n📁 База данных: {self.db_dir}")

        if not self.data_dir.exists():
            print(f"\n❌ Папка {self.data_dir} не существует!")
            exit(1)

        if rebuild_db and self.db_dir.exists():
            print("\n🗑️ Удаление старого индекса...")
            import shutil
            shutil.rmtree(self.db_dir)
            print(" ✅ Удалён")

        self._load_or_create_knowledge_base()

        print("\n🤖 Инициализация языковой модели...")
        self.llm = ChatOllama(model="llama3.1", temperature=0.3)
        print(" ✅ llama3.1 готова")

        print("\n" + "=" * 70)
        print("✅ СИСТЕМА ГОТОВА!")
        print("=" * 70)

    # ---------- Работа с базой знаний ----------

    def _load_or_create_knowledge_base(self):
        """Загрузка или создание FAISS индекса под enhanced_dataset."""
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        if self.db_dir.exists() and (self.db_dir / "index.faiss").exists():
            print("\n📚 Найден существующий FAISS индекс")
            print(f" Путь: {self.db_dir}")
            try:
                print(" ⏳ Загрузка...")
                self.vectorstore = FAISS.load_local(
                    str(self.db_dir),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                # Быстрая проверка
                _ = self.vectorstore.similarity_search("тест", k=1)
                print(" ✅ Индекс загружен успешно")
                return
            except Exception as e:
                print(f" ⚠️ Ошибка: {e}")
                print(" 🔄 Перестраиваем индекс...")

        print("\n📚 Создание нового FAISS индекса (enhanced_dataset)")
        print(" ⏳ Это может занять 5–15 минут\n")
        self._create_new_database(embeddings)

    def _create_new_database(self, embeddings):
        """Создание FAISS индекса из enhanced_dataset."""
        print("1️⃣ Загрузка документов...")

        documents = []
        json_files = list(self.data_dir.glob("*.json"))

        if not json_files:
            print(" ❌ В папке enhanced_dataset нет JSON‑файлов!")
            exit(1)

        total = len(json_files)
        print(f" Найдено: {total} файлов")

        for i, json_file in enumerate(json_files, 1):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                title = (data.get("title") or "").strip()

                # Основные разделы
                sections = data.get("sections", {})
                full_text = f"# {title}\n\n"

                for section_name, section_text in sections.items():
                    if not section_text or not str(section_text).strip():
                        continue
                    readable_name = section_name.replace("_", " ").title()
                    full_text += f"## {readable_name}\n{section_text}\n\n"

                if len(full_text) <= 100:
                    # Слишком короткий документ – пропускаем
                    continue

                # Метаданные из enhance_medical_dataset
                meta = data.get("metadata", {}) or {}
                doc_metadata = {
                    "title": title,
                    "disease": title,
                    "file": json_file.name,
                    "categories": meta.get("categories", []),
                    "symptoms": meta.get("symptoms", []),
                    "complexity": meta.get("complexity", ""),
                    "symptoms_count": meta.get("symptoms_count", 0),
                    "filled_sections": meta.get("filled_sections", 0),
                    "total_sections": meta.get("total_sections", 0),
                    "completeness_score": meta.get("completeness_score", 0.0),
                }

                documents.append(
                    Document(page_content=full_text, metadata=doc_metadata)
                )

                if i % 50 == 0 or i == total:
                    print(f" 📊 Обработано файлов: {i}/{total}")

            except Exception as e:
                print(f" ⚠️ Ошибка в {json_file.name}: {e}")

        print(f" ✅ Загружено заболеваний: {len(documents)}")

        # 2. Разбивка текста
        print("\n2️⃣ Разбивка текста на фрагменты...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )
        splits = text_splitter.split_documents(documents)
        total_splits = len(splits)
        print(f" ✅ Фрагментов: {total_splits}")

        # 3. Создание эмбеддингов
        print("\n3️⃣ Создание векторных эмбеддингов...")
        print(" ⏳ Это займёт время, дождитесь окончания\n")

        batch_size = 100
        vectorstore = None

        try:
            for i in range(0, total_splits, batch_size):
                batch = splits[i : i + batch_size]
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)

                progress = min(i + batch_size, total_splits)
                percent = (progress / total_splits) * 100
                print(f" 📊 {progress}/{total_splits} ({percent:.1f}%)")

            self.vectorstore = vectorstore

            # 4. Сохранение
            print("\n4️⃣ Сохранение индекса...")
            self.db_dir.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(self.db_dir))
            print(f" ✅ Сохранено в: {self.db_dir}")

        except Exception as e:
            print(f"\n ❌ Ошибка при создании индекса: {e}")
            import traceback

            traceback.print_exc()
            raise

    def _search_context(self, query: str, k: int = 3) -> str:
        """Поиск релевантного контекста в FAISS."""
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return "\n\n".join(doc.page_content[:700] for doc in docs)
        except Exception as e:
            print(f"⚠️ Ошибка поиска: {e}")
            return ""

    # ---------- Логика диалога ----------

    def _generate_question(self) -> str:
        """Генерация следующего вопроса врачу‑ботом."""
        search_query = f"{self.collected_info['chief_complaint']} " \
                       f"{' '.join(self.collected_info['symptoms'])}"
        context = self._search_context(search_query, k=2)

        history = "\n".join(
            f"{'Врач' if m['role'] == 'assistant' else 'Пациент'}: {m['content']}"
            for m in self.conversation_history[-4:]
        )

        prompt = ChatPromptTemplate.from_template(
            """
Ты врач, собирающий анамнез. Твоя задача узнать как можно больше информации о пациенте относительно его здоровья,
для того чтобы заполнить. В него входит несколько пунктов: Общее состояние здоровья пациента, какие у него симптомы, какие лекарства он принемает сейча
и как принимал в ближайщее время до этого.

ИСТОРИЯ:
{history}

ИНФОРМАЦИЯ:
- Жалоба: {chief_complaint}
- Симптомы: {symptoms}

КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ:
{context}

Задай ОДИН короткий уточняющий вопрос.
Вопрос:"""
        )

        try:
            from langchain_core.runnables import RunnableConfig

            response = self.llm.invoke(
                prompt.format(
                    history=history,
                    chief_complaint=self.collected_info["chief_complaint"]
                    or "не указано",
                    symptoms=", ".join(self.collected_info["symptoms"])
                    if self.collected_info["symptoms"]
                    else "нет",
                    context=context or "Нет данных",
                ),
                config=RunnableConfig(max_concurrency=1, timeout=30),
            )
            return response.content.strip()
        except Exception as e:
            print(f"\n⚠️ Ошибка LLM: {e}")
            fallback_questions = [
                "Как давно у вас эти симптомы?",
                "Усиливаются ли симптомы после еды?",
                "Есть ли температура?",
                "Была ли рвота?",
                "Где именно локализуется боль?",
            ]
            import random

            return random.choice(fallback_questions)

    def _extract_info(self, text: str):
        """Грубое извлечение симптомов и длительности из ответа пациента."""
        text_lower = text.lower()

        time_words = ["день", "дня", "дней", "неделю", "месяц", "год"]
        if any(w in text_lower for w in time_words) and not self.collected_info[
            "duration"
        ]:
            self.collected_info["duration"] = text

        symptoms_vocab = [
            "боль",
            "температура",
            "тошнота",
            "рвота",
            "слабость",
            "кашель",
            "насморк",
            "горло",
            "голова",
            "живот",
        ]
        for symptom in symptoms_vocab:
            if symptom in text_lower:
                if symptom not in " ".join(
                    self.collected_info["symptoms"]
                ).lower():
                    self.collected_info["symptoms"].append(symptom)

    def _should_continue(self) -> bool:
        """Решение, продолжать ли интервью."""
        questions = len(
            [m for m in self.conversation_history if m["role"] == "assistant"]
        )
        
        # Обязательные условия для остановки:
        # 1. Должно быть жалоба
        # 2. Минимум 3-4 симптома И длительность
        has_enough_info = (
            bool(self.collected_info["chief_complaint"]) and
            len(self.collected_info["symptoms"]) >= 3 and  # не 2, а 3+
            bool(self.collected_info["duration"])  # обязательно должна быть длительность
        )
        
        # Максимум 15 вопросов, но продолжаем, если не хватает информации
        return questions < 15 and not has_enough_info


    def _generate_report(self) -> str:
        """Генерация развёрнутого медицинского отчёта с анамнезом."""
        
        search_query = " ".join([
            self.collected_info["chief_complaint"],
            *self.collected_info["symptoms"]
        ])
        context = self._search_context(search_query, k=5)
        
        conversation = "\n".join([
            f"{'Врач' if m['role'] == 'assistant' else 'Пациент'}: {m['content']}"
            for m in self.conversation_history
        ])
        
        prompt = ChatPromptTemplate.from_template("""
Ты опытный врач, готовящий детальный анамнез пациента для коллег.
На основе диалога ниже, заполни структурированный медицинский отчёт.

ДИАЛОГ С ПАЦИЕНТОМ:
{conversation}

КЛИНИЧЕСКИЕ ДАННЫЕ ИЗ БАЗЫ:
{context}

СОБРАННАЯ ИНФОРМАЦИЯ:
- Основная жалоба: {chief_complaint}
- Симптомы: {symptoms}
- Длительность: {duration}
- Дополнительно: {additional_info}

Заполни подробный СТРУКТУРИРОВАННЫЙ АНАМНЕЗ для врача:

**ANAMNESIS VITAE (История жизни):**
[Заполни на основе разговора: возраст, пол (если упомянут), профессиональные вредности, 
образ жизни, курение/алкоголь, хронические заболевания, алергии]

**ANAMNESIS MORBI (История болезни):**
[Развернуто опиши: начало заболевания, течение, развитие симптомов, факторы, усиливающие/ослабляющие]

**ЖАЛОБЫ И СИМПТОМЫ:**
[Подробное описание каждого симптома: характер, интенсивность, локализация, время появления]

**ДИФФЕРЕНЦИАЛЬНЫЙ ДИАГНОЗ:**
[На основе клинических данных и симптомов выдвини 3-5 наиболее вероятных диагнозов с обоснованием]

**ПЛАН ОБСЛЕДОВАНИЯ:**
[Перечисли необходимые анализы и исследования, специалистов]

**ПРИМЕЧАНИЯ ДЛЯ ВРАЧА:**
[Ключевые моменты, на которые обратить внимание при осмотре]

Готовый отчёт:""")
    
        try:
            from langchain_core.runnables import RunnableConfig
            
            print(" ⏳ Генерация подробного отчёта (30-60 секунд)...")
            response = self.llm.invoke(
                prompt.format(
                    conversation=conversation,
                    context=context or "Требуется дополнительное обследование",
                    chief_complaint=self.collected_info["chief_complaint"] or "не указана",
                    symptoms=", ".join(self.collected_info["symptoms"]) if self.collected_info["symptoms"] else "не указаны",
                    duration=self.collected_info["duration"] or "не указана",
                    additional_info=", ".join(self.collected_info["additional_info"]) if self.collected_info["additional_info"] else "отсутствует"
                ),
                config=RunnableConfig(
                    timeout=90  # Больше времени на подробный отчёт
                ),
            )
            return response.content
            
        except Exception as e:
            print(f"\n⚠️ Ошибка генерации: {e}")
            
            # Fallback - структурированный отчёт вручную
            return f"""
**ANAMNESIS VITAE:**
Информация об истории жизни не была собрана в ходе первичного интервью.

**ANAMNESIS MORBI:**
Пациент обратился с основной жалобой на: {self.collected_info['chief_complaint']}

Начало заболевания: {self.collected_info['duration'] if self.collected_info['duration'] else 'время начала не уточнено'}

**ЖАЛОБЫ И СИМПТОМЫ:**
{chr(10).join(f"- {s.capitalize()}" for s in self.collected_info['symptoms']) if self.collected_info['symptoms'] else "- Симптомы не указаны"}

**ДИФФЕРЕНЦИАЛЬНЫЙ ДИАГНОЗ:**
На основе предъявленных жалоб и симптомов необходимо рассмотреть:
- Острые инфекционные заболевания
- Хронические системные заболевания
- Функциональные расстройства
- Психосоматические нарушения

Требуется уточняющее обследование.

**ПЛАН ОБСЛЕДОВАНИЯ:**
1. Общий анализ крови (ОАК)
2. Общий анализ мочи (ОАМ)
3. Биохимический анализ крови
4. УЗИ органов брюшной полости (при наличии жалоб на боли в животе)
5. ЭКГ (при наличии жалоб на боли в груди или нарушение ритма)
6. По показаниям - консультация специалистов

**ПРИМЕЧАНИЯ ДЛЯ ВРАЧА:**
- Требуется более детальное уточнение анамнеза
- Необходима физикальная диагностика
- На основе результатов анализов корректировать дифференциальный диагноз
"""

    # ---------- Запуск интервью ----------

    def start_interview(self):
        print("\n" + "=" * 70)
        print("🩺 МЕДИЦИНСКОЕ ИНТЕРВЬЮ")
        print("=" * 70)
        print("\nКоманды: 'стоп' — завершить, 'exit' — выход\n")

        greeting = "Здравствуйте! Что вас беспокоит?"
        print(f"🤖: {greeting}\n")
        self.conversation_history.append({"role": "assistant", "content": greeting})

        complaint = input("👤: ").strip()
        if complaint.lower() in ["exit", "выход"]:
            print("\n👋 До свидания!")
            return
        if not complaint:
            print("⚠️ Введите жалобу")
            return

        self.collected_info["chief_complaint"] = complaint
        self.conversation_history.append({"role": "user", "content": complaint})
        self._extract_info(complaint)

        while self._should_continue():
            try:
                question = self._generate_question()
                print(f"\n🤖: {question}\n")
                self.conversation_history.append(
                    {"role": "assistant", "content": question}
                )

                answer = input("👤: ").strip()
                if answer.lower() in ["exit", "выход"]:
                    print("\n👋 До свидания!")
                    return
                if answer.lower() == "стоп":
                    break
                if not answer:
                    continue

                self.conversation_history.append(
                    {"role": "user", "content": answer}
                )
                self._extract_info(answer)
            except Exception as e:
                print(f"⚠️ Ошибка: {e}")
                break

        print("\n" + "=" * 70)
        print("📋 ГЕНЕРАЦИЯ ОТЧЁТА...")
        print("=" * 70)

        try:
            report = self._generate_report()
            print("\n" + "=" * 70)
            print("📄 МЕДИЦИНСКИЙ ОТЧЁТ")
            print("=" * 70 + "\n")
            print(report)
            print("\n" + "=" * 70)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.script_dir / f"report_{timestamp}.txt"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write("МЕДИЦИНСКИЙ ОТЧЁТ\n")
                f.write(
                    f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
                )
                f.write("=" * 70 + "\n\n")
                f.write(report)

            print(f"\n💾 Сохранено в файл: {report_file.name}")
        except Exception as e:
            print(f"❌ Ошибка при сохранении отчёта: {e}")


if __name__ == "__main__":
    import sys

    rebuild = "--rebuild" in sys.argv
    try:
        bot = MedicalInterviewBot(rebuild_db=rebuild)
        bot.start_interview()
    except KeyboardInterrupt:
        print("\n\n👋 Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()
