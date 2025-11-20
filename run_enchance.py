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
        self.data_dir = self.script_dir / "enhanced_dataset"

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
        print("🏥 МЕДИЦИНСКИЙ ИНТЕРВЬЮЕР v3.0 (enhanced_dataset)")
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
                sections = data.get("sections", {})
                full_text = f"# {title}\n\n"

                for section_name, section_text in sections.items():
                    if not section_text or not str(section_text).strip():
                        continue
                    readable_name = section_name.replace("_", " ").title()
                    full_text += f"## {readable_name}\n{section_text}\n\n"

                if len(full_text) <= 100:
                    continue

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

        print("\n2️⃣ Разбивка текста на фрагменты...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )
        splits = text_splitter.split_documents(documents)
        total_splits = len(splits)
        print(f" ✅ Фрагментов: {total_splits}")

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

    # ---------- Валидация ввода ----------

    def _is_valid_medical_input(self, text: str) -> bool:
        """Проверяет валидность медицинского ввода."""
        if not text or len(text.strip()) < 3:
            return False
        
        text_lower = text.lower()
        
        # Чёрный список бессмыслицы
        bad_words = ["не стоит", "ха ха", "кек", "zzz", "123", "ненужно"]
        if any(word in text_lower for word in bad_words):
            return False
        
        # Минимум 2 слова
        return len(text.split()) >= 2

    def _get_valid_patient_answer(self) -> str:
        """Получает ответ пациента с валидацией."""
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts:
            answer = input("👤: ").strip()
            
            # Обработка команд
            if answer.lower() in ["exit", "выход"]:
                print("\n👋 До свидания!")
                return None
            if answer.lower() == "стоп":
                return "STOP"
            
            # Проверка валидности
            if not self._is_valid_medical_input(answer):
                attempts += 1
                if attempts < max_attempts:
                    print(f"\n⚠️ Пожалуйста, опишите медицинскую проблему. "
                          f"Попытка {attempts}/{max_attempts}\n")
                    continue
                else:
                    print("\n❌ Похоже, вы не хотите обсуждать медицинскую проблему.")
                    return None
            
            return answer
        
        return None

    # ---------- Логика диалога ----------

    def _should_continue(self) -> bool:
        """Решение, продолжать ли интервью."""
        questions = len(
            [m for m in self.conversation_history if m["role"] == "assistant"]
        )
        
        # Проверка условий остановки
        has_chief_complaint = bool(self.collected_info["chief_complaint"])
        has_symptoms = len(self.collected_info["symptoms"]) >= 3
        has_duration = bool(self.collected_info["duration"])
        
        has_enough_info = has_chief_complaint and has_symptoms and has_duration
        
        # Для отладки
        print(f"\n📊 Статус: вопросов={questions}, симптомов={len(self.collected_info['symptoms'])}, "
              f"длительность={'✓' if has_duration else '✗'}")
        
        # Продолжаем, если не задали 15 вопросов И не собрали информацию
        return questions < 15 and not has_enough_info

    def _generate_question(self) -> str:
        """Генерация следующего вопроса врачу-ботом."""
        search_query = f"{self.collected_info['chief_complaint']} " \
                       f"{' '.join(self.collected_info['symptoms'])}"
        context = self._search_context(search_query, k=2)

        history = "\n".join(
            f"{'Врач' if m['role'] == 'assistant' else 'Пациент'}: {m['content']}"
            for m in self.conversation_history[-4:]
        )

        prompt = ChatPromptTemplate.from_template(
            """
Ты врач, собирающий анамнез. Если пациент ответил невразумительно, 
тактично переведи разговор на медицинскую проблему.

ИСТОРИЯ:
{history}

ИНФОРМАЦИЯ:
- Жалоба: {chief_complaint}
- Симптомы: {symptoms}

КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ:
{context}

Задай ОДИН короткий уточняющий вопрос по делу:"""
        )

        try:
            from langchain_core.runnables import RunnableConfig

            response = self.llm.invoke(
                prompt.format(
                    history=history,
                    chief_complaint=self.collected_info["chief_complaint"] or "не указано",
                    symptoms=", ".join(self.collected_info["symptoms"]) if self.collected_info["symptoms"] else "нет",
                    context=context or "Нет данных",
                ),
                config=RunnableConfig(max_concurrency=1, timeout=60),
            )
            return response.content.strip()
        except Exception as e:
            print(f"\n⚠️ Ошибка LLM: {e}")
            fallback_questions = [
                "Как давно у вас эти симптомы?",
                "Усиливаются ли симптомы после еды или физической нагрузки?",
                "Есть ли температура?",
                "Где именно локализуется боль?",
                "Есть ли тошнота или рвота?",
            ]
            import random
            return random.choice(fallback_questions)

    def _extract_info(self, text: str):
        """Грубое извлечение симптомов и длительности из ответа пациента."""
        text_lower = text.lower()

        time_words = ["день", "дня", "дней", "неделю", "месяц", "год", "час", "часов", "минут"]
        if any(w in text_lower for w in time_words) and not self.collected_info["duration"]:
            self.collected_info["duration"] = text

        symptoms_vocab = [
            "боль", "температура", "тошнота", "рвота", "слабость", 
            "кашель", "насморк", "горло", "голова", "живот", 
            "сыпь", "озноб", "головокружение", "диарея", "запор"
        ]
        
        for symptom in symptoms_vocab:
            if symptom in text_lower:
                if symptom not in " ".join(self.collected_info["symptoms"]).lower():
                    self.collected_info["symptoms"].append(symptom)

    def _generate_report(self) -> str:
        """Генерация развёрнутого медицинского отчёта с валидацией."""
        
        # Проверка основной жалобы
        if not self.collected_info["chief_complaint"]:
            return """❌ ОШИБКА: Основная жалоба пациента не была собрана.
        
Невозможно создать медицинский отчёт без основной информации о проблеме пациента.
Пожалуйста, перезапустите интервью."""
        
        # Санитизация истории
        clean_history = []
        for msg in self.conversation_history:
            content = msg.get("content", "").strip().lower()
            
            if any(bad in content for bad in [
                "не могу ответить",
                "неуместный вопрос",
                "не является медицинской",
                "не стоит",
                "ха ха",
                "кек"
            ]):
                continue
            
            clean_history.append(msg)
        
        if len(clean_history) < 3:
            clean_history = self.conversation_history
        
        conversation = "\n".join([
            f"{'Врач' if m['role'] == 'assistant' else 'Пациент'}: {m['content']}"
            for m in clean_history
        ])
        
        search_query = " ".join([
            self.collected_info["chief_complaint"],
            *self.collected_info["symptoms"]
        ])
        context = self._search_context(search_query, k=5)
        
        # Валидация симптомов
        symptoms_list = [s for s in self.collected_info["symptoms"] if len(s) > 2]
        if not symptoms_list:
            symptoms_list = ["не уточнены"]
        
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

Заполни подробный СТРУКТУРИРОВАННЫЙ АНАМНЕЗ для врача:

**ANAMNESIS MORBI (История болезни):**
[Развернуто опиши: начало заболевания, течение, развитие симптомов]

**ЖАЛОБЫ И СИМПТОМЫ:**
[Подробное описание каждого симптома]

**ДИФФЕРЕНЦИАЛЬНЫЙ ДИАГНОЗ:**
[На основе жалобы и симптомов выдвини 3-5 вероятных диагнозов]

**ПЛАН ОБСЛЕДОВАНИЯ:**
1. Общий анализ крови (ОАК)
2. Биохимический анализ крови
3. [Дополнительные исследования по показаниям]

**РЕКОМЕНДАЦИИ:**
[Общие рекомендации пациенту]

Отчёт:""")
        
        try:
            from langchain_core.runnables import RunnableConfig
            
            print(" ⏳ Генерация отчёта (30-60 секунд)...")
            response = self.llm.invoke(
                prompt.format(
                    conversation=conversation if conversation else "Диалог не был продуктивен",
                    context=context or "Требуется дополнительное обследование",
                    chief_complaint=self.collected_info["chief_complaint"],
                    symptoms=", ".join(symptoms_list),
                    duration=self.collected_info["duration"] or "не указана"
                ),
                config=RunnableConfig(timeout=90),
            )
            return response.content
            
        except Exception as e:
            print(f"\n⚠️ Ошибка генерации: {e}")
            
            # Fallback отчёт
            return f"""
**ANAMNESIS MORBI:**
Пациент обратился с жалобой на: {self.collected_info['chief_complaint']}
Длительность: {self.collected_info['duration'] if self.collected_info['duration'] else 'не указана'}

**ЖАЛОБЫ И СИМПТОМЫ:**
{chr(10).join(f"- {s.capitalize()}" for s in symptoms_list)}

**ДИФФЕРЕНЦИАЛЬНЫЙ ДИАГНОЗ:**
На основе предъявленных жалоб необходимо рассмотреть:
- Острые инфекционные заболевания
- Хронические системные заболевания  
- Функциональные расстройства

**ПЛАН ОБСЛЕДОВАНИЯ:**
1. Общий анализ крови (ОАК)
2. Общий анализ мочи (ОАМ)
3. Биохимический анализ крови
4. УЗИ по показаниям
5. По результатам — консультация узких специалистов

**РЕКОМЕНДАЦИИ:**
- Соблюдение режима покоя
- Обильное питьё
- Повторная консультация при ухудшении состояния
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

        # Получение жалобы пациента с валидацией
        complaint = self._get_valid_patient_answer()
        if complaint is None:
            return
        if complaint == "STOP":
            print("\n👋 До свидания!")
            return

        self.collected_info["chief_complaint"] = complaint
        self.conversation_history.append({"role": "user", "content": complaint})
        self._extract_info(complaint)

        # Основной цикл интервью
        while self._should_continue():
            try:
                question = self._generate_question()
                print(f"\n🤖: {question}\n")
                self.conversation_history.append(
                    {"role": "assistant", "content": question}
                )

                # Валидированный ввод пациента
                answer = self._get_valid_patient_answer()
                if answer is None:
                    break
                if answer == "STOP":
                    break

                self.conversation_history.append({"role": "user", "content": answer})
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
                f.write(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
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