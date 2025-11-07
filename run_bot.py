from pathlib import Path
import json
import shutil
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

class MedicalInterviewBot:
    def __init__(self, rebuild_db: bool = False):
        self.script_dir = Path(__file__).parent
        self.data_dir = self.script_dir / "cleaned_dataset"
        self.db_dir = self.script_dir / "vector_db"
        
        self.conversation_history = []
        self.collected_info = {
            "chief_complaint": "",
            "symptoms": [],
            "duration": "",
            "additional_info": []
        }
        
        print("=" * 70)
        print("🏥 МЕДИЦИНСКИЙ ИНТЕРВЬЮЕР v2.0")
        print("=" * 70)
        
        # Проверяем датасет
        if not self.data_dir.exists() or not list(self.data_dir.glob("*.json")):
            print(f"\n❌ Папка {self.data_dir} пуста или не существует!")
            print("   Создайте датасет с помощью create_clean_dataset.py")
            exit(1)
        
        # Удаляем повреждённую базу если нужно
        if rebuild_db and self.db_dir.exists():
            print("\n🗑️ Удаление старой базы данных...")
            try:
                shutil.rmtree(self.db_dir)
                print("   ✅ Старая база удалена")
            except Exception as e:
                print(f"   ⚠️ Ошибка удаления: {e}")
        
        # Загружаем или создаём базу
        self._load_or_create_knowledge_base()
        
        # Инициализация LLM
        print("\n🤖 Инициализация языковой модели...")
        self.llm = ChatOllama(model="llama3.2", temperature=0.3)
        print("   ✅ Готова")
        
        print("\n" + "=" * 70)
        print("✅ СИСТЕМА ГОТОВА!")
        print("=" * 70)
    
    def _load_or_create_knowledge_base(self):
        """Умная загрузка: используем кеш или создаём новый"""
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Проверяем существует ли база
        db_exists = self.db_dir.exists() and any(self.db_dir.iterdir())
        
        if db_exists:
            print("\n📚 Найдена существующая векторная база")
            print(f"   Путь: {self.db_dir}")
            
            try:
                # Пытаемся загрузить существующую базу
                self.vectorstore = Chroma(
                    persist_directory=str(self.db_dir),
                    embedding_function=embeddings
                )
                
                # Проверяем что база работает
                test_results = self.vectorstore.similarity_search("тест", k=1)
                
                print("   ✅ База загружена успешно")
                print(f"   📊 Содержит документы: {len(test_results) > 0}")
                return
                
            except Exception as e:
                print(f"   ⚠️ Ошибка загрузки базы: {e}")
                print("   🔄 Пересоздаём базу данных...")
                
                # Удаляем повреждённую базу
                try:
                    shutil.rmtree(self.db_dir)
                except:
                    pass
        
        # Создаём новую базу
        print("\n📚 Создание новой векторной базы")
        print("   ⏳ Это займёт 2-5 минут (делается только один раз)\n")
        
        self._create_new_database(embeddings)
    
    def _create_new_database(self, embeddings):
        """Создание новой векторной базы с нуля"""
        
        # 1. Загрузка документов
        print("1️⃣ Загрузка медицинских документов...")
        documents = []
        json_files = list(self.data_dir.glob("*.json"))
        
        if not json_files:
            print("   ❌ Не найдено JSON файлов в cleaned_dataset/")
            exit(1)
        
        print(f"   Найдено файлов: {len(json_files)}")
        
        for i, json_file in enumerate(json_files, 1):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                title = data.get("title", "")
                
                # Собираем текст из секций
                full_text = f"# {title}\n\n"
                
                if "sections" in data:
                    for section_name, section_text in data["sections"].items():
                        if section_text and str(section_text).strip():
                            readable_name = section_name.replace("_", " ").title()
                            full_text += f"## {readable_name}\n{section_text}\n\n"
                
                if full_text.strip() and len(full_text) > 100:
                    doc = Document(
                        page_content=full_text,
                        metadata={
                            "title": title,
                            "disease": title,
                            "source": json_file.name
                        }
                    )
                    documents.append(doc)
                    
                    if i % 10 == 0:
                        print(f"   Обработано: {i}/{len(json_files)}")
                        
            except Exception as e:
                print(f"   ⚠️ Ошибка в {json_file.name}: {e}")
        
        if not documents:
            print("   ❌ Не удалось загрузить документы!")
            exit(1)
        
        print(f"   ✅ Загружено заболеваний: {len(documents)}")
        
        # 2. Разбивка на чанки
        print("\n2️⃣ Разбивка текста на фрагменты...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        print(f"   ✅ Создано фрагментов: {len(splits)}")
        
        # 3. Создание векторной базы
        print("\n3️⃣ Создание векторных индексов...")
        print("   ⏳ Подождите, это может занять несколько минут...")
        
        try:
            # Создаём базу порциями для надёжности
            batch_size = 50
            self.db_dir.mkdir(exist_ok=True)
            
            for i in range(0, len(splits), batch_size):
                batch = splits[i:i+batch_size]
                
                if i == 0:
                    # Первая порция - создаём базу
                    self.vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=str(self.db_dir)
                    )
                else:
                    # Остальные порции - добавляем
                    self.vectorstore.add_documents(batch)
                
                progress = min(i + batch_size, len(splits))
                print(f"   📊 Прогресс: {progress}/{len(splits)} фрагментов")
            
            print("\n   ✅ Векторная база создана и сохранена!")
            print(f"   📁 Путь: {self.db_dir}")
            
        except Exception as e:
            print(f"\n   ❌ Ошибка создания базы: {e}")
            
            # Удаляем повреждённую базу
            if self.db_dir.exists():
                try:
                    shutil.rmtree(self.db_dir)
                except:
                    pass
            
            raise
    
    def _search_context(self, query: str, k: int = 3) -> str:
        """Поиск релевантного контекста"""
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content[:700] for doc in docs])
            return context
        except Exception as e:
            print(f"⚠️ Ошибка поиска: {e}")
            return ""
    
    def _generate_question(self) -> str:
        """Генерация следующего вопроса"""
        
        # Контекст из БД
        search_query = f"{self.collected_info['chief_complaint']} {' '.join(self.collected_info['symptoms'])}"
        context = self._search_context(search_query, k=2)
        
        # История
        history = "\n".join([
            f"{'Врач' if msg['role'] == 'assistant' else 'Пациент'}: {msg['content']}"
            for msg in self.conversation_history[-4:]
        ])
        
        prompt = ChatPromptTemplate.from_template("""
Ты врач, собирающий анамнез у пациента.

ИСТОРИЯ РАЗГОВОРА:
{history}

СОБРАННАЯ ИНФОРМАЦИЯ:
- Жалоба: {chief_complaint}
- Симптомы: {symptoms}
- Длительность: {duration}

КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ:
{context}

Задай ОДИН короткий вопрос для уточнения:
1. Характер симптомов (острая/тупая боль, где именно)
2. Длительность и динамика
3. Связанные симптомы из клинических рекомендаций
4. Провоцирующие факторы

Вопрос должен быть понятным пациенту.

Вопрос:""")
        
        try:
            response = self.llm.invoke(prompt.format(
                history=history,
                chief_complaint=self.collected_info["chief_complaint"] or "не указано",
                symptoms=", ".join(self.collected_info["symptoms"]) if self.collected_info["symptoms"] else "нет",
                duration=self.collected_info["duration"] or "не указано",
                context=context if context else "Нет релевантной информации"
            ))
            return response.content.strip()
        except Exception as e:
            print(f"⚠️ Ошибка генерации вопроса: {e}")
            return "Расскажите подробнее о ваших симптомах?"
    
    def _extract_info(self, text: str):
        """Извлечение информации из ответа"""
        text_lower = text.lower()
        
        # Длительность
        time_words = ['день', 'дня', 'дней', 'неделю', 'недели', 'месяц', 'год']
        if any(w in text_lower for w in time_words) and not self.collected_info["duration"]:
            self.collected_info["duration"] = text
        
        # Симптомы
        symptoms = ['боль', 'температура', 'жар', 'тошнота', 'рвота', 'слабость',
                   'головная', 'кашель', 'одышка', 'диарея', 'запор', 'зуд', 'отек']
        
        for symptom in symptoms:
            if symptom in text_lower:
                if symptom not in " ".join(self.collected_info["symptoms"]).lower():
                    self.collected_info["symptoms"].append(symptom)
    
    def _should_continue(self) -> bool:
        """Проверка нужно ли продолжать"""
        questions = len([m for m in self.conversation_history if m["role"] == "assistant"])
        
        has_info = (
            bool(self.collected_info["chief_complaint"]) and
            (len(self.collected_info["symptoms"]) >= 2 or bool(self.collected_info["duration"]))
        )
        
        return questions < 8 and not has_info
    
    def _generate_report(self) -> str:
        """Генерация медицинского отчёта"""
        
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
Составь структурированный медицинский отчёт для врача.

БЕСЕДА С ПАЦИЕНТОМ:
{conversation}

КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ:
{context}

Формат отчёта:

**Anamnesis morbi:**
[История текущего заболевания: жалобы, характер, длительность]

**Anamnesis vitae:**
[Анамнез жизни, если упоминался]

**Clinical data:**
[Доступные данные или их отсутствие]

**Differential diagnosis:**
[Возможные диагнозы на основе клинических рекомендаций]

**Recommendations:**
[План обследования и тактика]

Отчёт:""")
        
        try:
            response = self.llm.invoke(prompt.format(
                conversation=conversation,
                context=context if context else "Требуется дополнительное обследование"
            ))
            return response.content
        except Exception as e:
            print(f"⚠️ Ошибка генерации отчёта: {e}")
            return "Ошибка генерации отчёта"
    
    def start_interview(self):
        """Запуск интервью"""
        print("\n" + "=" * 70)
        print("🩺 НАЧАЛО МЕДИЦИНСКОГО ИНТЕРВЬЮ")
        print("=" * 70)
        print("\nКоманды: 'стоп' - завершить, 'exit' - выход\n")
        
        # Приветствие
        greeting = "Здравствуйте! Расскажите, что вас беспокоит?"
        print(f"🤖: {greeting}\n")
        self.conversation_history.append({"role": "assistant", "content": greeting})
        
        # Основная жалоба
        complaint = input("👤: ").strip()
        
        if complaint.lower() in ['exit', 'выход']:
            print("\n👋 До свидания!")
            return
        
        if not complaint or complaint.lower() == 'стоп':
            print("⚠️ Введите вашу жалобу")
            return
        
        self.collected_info["chief_complaint"] = complaint
        self.conversation_history.append({"role": "user", "content": complaint})
        self._extract_info(complaint)
        
        # Цикл вопросов
        while self._should_continue():
            try:
                question = self._generate_question()
                print(f"\n🤖: {question}\n")
                self.conversation_history.append({"role": "assistant", "content": question})
                
                answer = input("👤: ").strip()
                
                if answer.lower() in ['exit', 'выход']:
                    print("\n👋 До свидания!")
                    return
                
                if answer.lower() == 'стоп':
                    break
                
                if not answer:
                    continue
                
                self.conversation_history.append({"role": "user", "content": answer})
                self._extract_info(answer)
                
            except Exception as e:
                print(f"⚠️ Ошибка: {e}")
                break
        
        # Отчёт
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
            
            # Сохранение
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.script_dir / f"report_{timestamp}.txt"
            
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(f"МЕДИЦИНСКИЙ ОТЧЁТ\n")
                f.write(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
                f.write("=" * 70 + "\n\n")
                f.write(report)
            
            print(f"\n💾 Сохранено: {report_file.name}")
            
        except Exception as e:
            print(f"❌ Ошибка генерации отчёта: {e}")

if __name__ == "__main__":
    import sys
    
    # Поддержка флага --rebuild
    rebuild = "--rebuild" in sys.argv or "-r" in sys.argv
    
    try:
        bot = MedicalInterviewBot(rebuild_db=rebuild)
        bot.start_interview()
    except KeyboardInterrupt:
        print("\n\n👋 Прервано")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
