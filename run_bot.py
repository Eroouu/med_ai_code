from pathlib import Path
import json
import pickle
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

class MedicalInterviewBot:
    def __init__(self, rebuild_db: bool = False):
        self.script_dir = Path(__file__).parent
        self.data_dir = self.script_dir / "cleaned_dataset"
        self.db_file = self.script_dir / "faiss_index.pkl"
        
        self.conversation_history = []
        self.collected_info = {
            "chief_complaint": "",
            "symptoms": [],
            "duration": "",
            "additional_info": []
        }
        
        print("=" * 70)
        print("🏥 МЕДИЦИНСКИЙ ИНТЕРВЬЮЕР v2.1 (FAISS)")
        print("=" * 70)
        
        if not self.data_dir.exists():
            print(f"\n❌ Папка {self.data_dir} не существует!")
            exit(1)
        
        # Удаляем старый индекс если rebuild
        if rebuild_db and self.db_file.exists():
            print("\n🗑️ Удаление старого индекса...")
            self.db_file.unlink()
            print("   ✅ Удалён")
        
        self._load_or_create_knowledge_base()
        
        print("\n🤖 Инициализация языковой модели...")
        self.llm = ChatOllama(model="llama3.1", temperature=0.3)
        print("   ✅ llama3.1 готова")
        
        print("\n" + "=" * 70)
        print("✅ СИСТЕМА ГОТОВА!")
        print("=" * 70)
    
    def _load_or_create_knowledge_base(self):
        """Загрузка или создание FAISS индекса"""
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Проверяем существует ли индекс
        if self.db_file.exists():
            print("\n📚 Найден существующий FAISS индекс")
            print(f"   Путь: {self.db_file}")
            
            try:
                with open(self.db_file, "rb") as f:
                    self.vectorstore = pickle.load(f)
                
                # Проверка работоспособности
                test = self.vectorstore.similarity_search("тест", k=1)
                
                print("   ✅ Индекс загружен успешно")
                return
                
            except Exception as e:
                print(f"   ⚠️ Ошибка загрузки: {e}")
                print("   🔄 Создаём новый индекс...")
        
        # Создание нового индекса
        print("\n📚 Создание нового FAISS индекса")
        print("   ⏳ Займёт 2-5 минут\n")
        
        self._create_new_database(embeddings)
    
    def _create_new_database(self, embeddings):
        """Создание нового FAISS индекса"""
        
        # 1. Загрузка документов
        print("1️⃣ Загрузка документов...")
        documents = []
        json_files = list(self.data_dir.glob("*.json"))
        
        if not json_files:
            print("   ❌ Нет JSON файлов!")
            exit(1)
        
        print(f"   Найдено: {len(json_files)} файлов")
        
        for i, json_file in enumerate(json_files, 1):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                title = data.get("title", "")
                full_text = f"# {title}\n\n"
                
                if "sections" in data:
                    for section_name, section_text in data["sections"].items():
                        if section_text and str(section_text).strip():
                            readable_name = section_name.replace("_", " ").title()
                            full_text += f"## {readable_name}\n{section_text}\n\n"
                
                if len(full_text) > 100:
                    doc = Document(
                        page_content=full_text,
                        metadata={"title": title, "disease": title}
                    )
                    documents.append(doc)
                    
                if i % 50 == 0:
                    print(f"   Обработано: {i}/{len(json_files)}")
                    
            except Exception as e:
                print(f"   ⚠️ {json_file.name}: {e}")
        
        print(f"   ✅ Загружено: {len(documents)} заболеваний")
        
        # 2. Разбивка
        print("\n2️⃣ Разбивка текста...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        splits = text_splitter.split_documents(documents)
        print(f"   ✅ Фрагментов: {len(splits)}")
        
        # 3. Создание FAISS индекса
        print("\n3️⃣ Создание FAISS индекса...")
        print("   ⏳ Подождите...")
        
        try:
            # FAISS создаётся за один раз - более стабильно
            self.vectorstore = FAISS.from_documents(splits, embeddings)
            
            # Сохраняем индекс
            print("\n4️⃣ Сохранение индекса...")
            with open(self.db_file, "wb") as f:
                pickle.dump(self.vectorstore, f)
            
            print(f"   ✅ Индекс сохранён: {self.db_file}")
            
        except Exception as e:
            print(f"\n   ❌ Ошибка: {e}")
            raise
    
    def _search_context(self, query: str, k: int = 3) -> str:
        """Поиск контекста"""
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content[:700] for doc in docs])
            return context
        except Exception as e:
            print(f"⚠️ Ошибка поиска: {e}")
            return ""
    
    def _generate_question(self) -> str:
        """Генерация вопроса"""
        search_query = f"{self.collected_info['chief_complaint']} {' '.join(self.collected_info['symptoms'])}"
        context = self._search_context(search_query, k=2)
        
        history = "\n".join([
            f"{'Врач' if m['role'] == 'assistant' else 'Пациент'}: {m['content']}"
            for m in self.conversation_history[-4:]
        ])
        
        prompt = ChatPromptTemplate.from_template("""
Ты врач, собирающий анамнез.

ИСТОРИЯ:
{history}

ИНФОРМАЦИЯ:
- Жалоба: {chief_complaint}
- Симптомы: {symptoms}

КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ:
{context}

Задай ОДИН короткий вопрос для уточнения симптомов.

Вопрос:""")
        
        try:
            response = self.llm.invoke(prompt.format(
                history=history,
                chief_complaint=self.collected_info["chief_complaint"] or "не указано",
                symptoms=", ".join(self.collected_info["symptoms"]) if self.collected_info["symptoms"] else "нет",
                context=context or "Нет данных"
            ))
            return response.content.strip()
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
            return "Расскажите подробнее?"
    
    def _extract_info(self, text: str):
        """Извлечение информации"""
        text_lower = text.lower()
        
        time_words = ['день', 'дня', 'дней', 'неделю', 'месяц', 'год']
        if any(w in text_lower for w in time_words) and not self.collected_info["duration"]:
            self.collected_info["duration"] = text
        
        symptoms = ['боль', 'температура', 'тошнота', 'рвота', 'слабость',
                   'кашель', 'насморк', 'горло', 'голова', 'живот']
        
        for symptom in symptoms:
            if symptom in text_lower:
                if symptom not in " ".join(self.collected_info["symptoms"]).lower():
                    self.collected_info["symptoms"].append(symptom)
    
    def _should_continue(self) -> bool:
        """Проверка продолжения"""
        questions = len([m for m in self.conversation_history if m["role"] == "assistant"])
        has_info = (
            bool(self.collected_info["chief_complaint"]) and
            (len(self.collected_info["symptoms"]) >= 2 or bool(self.collected_info["duration"]))
        )
        return questions < 8 and not has_info
    
    def _generate_report(self) -> str:
        """Генерация отчёта"""
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
Составь медицинский отчёт для врача.

БЕСЕДА:
{conversation}

КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ:
{context}

Формат:

**Anamnesis morbi:**
[История заболевания]

**Differential diagnosis:**
[Возможные диагнозы]

**Recommendations:**
[План обследования]

Отчёт:""")
        
        try:
            response = self.llm.invoke(prompt.format(
                conversation=conversation,
                context=context or "Требуется обследование"
            ))
            return response.content
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
            return "Ошибка генерации"
    
    def start_interview(self):
        """Запуск интервью"""
        print("\n" + "=" * 70)
        print("🩺 МЕДИЦИНСКОЕ ИНТЕРВЬЮ")
        print("=" * 70)
        print("\nКоманды: 'стоп' - завершить, 'exit' - выход\n")
        
        greeting = "Здравствуйте! Что вас беспокоит?"
        print(f"🤖: {greeting}\n")
        self.conversation_history.append({"role": "assistant", "content": greeting})
        
        complaint = input("👤: ").strip()
        
        if complaint.lower() in ['exit', 'выход']:
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
                f.write(f"МЕДИЦИНСКИЙ ОТЧЁТ\n")
                f.write(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
                f.write("=" * 70 + "\n\n")
                f.write(report)
            
            print(f"\n💾 Сохранено: {report_file.name}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    import sys
    rebuild = "--rebuild" in sys.argv
    
    try:
        bot = MedicalInterviewBot(rebuild_db=rebuild)
        bot.start_interview()
    except KeyboardInterrupt:
        print("\n\n👋 Прервано")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
