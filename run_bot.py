from pathlib import Path
import json
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

script_dir = Path(__file__).parent
data_dir = script_dir / "cleaned_dataset"  # Используем ОЧИЩЕННЫЕ данные!
db_dir = script_dir / "chroma_db_optimized"

db_dir.mkdir(exist_ok=True)

try:
    print("=" * 70)
    print("🚀 ИНИЦИАЛИЗАЦИЯ ОПТИМИЗИРОВАННОЙ RAG-СИСТЕМЫ")
    print("=" * 70)

    print("\n1️⃣ Загрузка очищенных данных...")
    documents = []
    json_files = list(data_dir.glob("**/*.json"))

    print(f"   Найдено {len(json_files)} файлов")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Собираем все текстовые содержимое
            full_text = data.get("title", "") + "\n\n"

            # Добавляем текст из разделов
            if "sections" in data:
                for section_name, section_data in data["sections"].items():
                    if isinstance(section_data, dict) and "text" in section_data:
                        full_text += f"## {section_name}\n{section_data['text']}\n\n"

            # Добавляем таблицы в текстовом формате
            if "applications" in data:
                for app_name, app_data in data["applications"].items():
                    if app_data.get("type") == "tables":
                        full_text += f"### {app_name} (таблица)\n"
                        for table in app_data.get("tables", []):
                            for row in table:
                                full_text += (
                                    " | ".join(str(cell) for cell in row) + "\n"
                                )
                            full_text += "\n"

            if full_text.strip():
                doc = Document(
                    page_content=full_text, metadata={"source": json_file.name}
                )
                documents.append(doc)
                print(f"   ✅ {json_file.name}")
        except Exception as e:
            print(f"   ⚠️  {json_file.name}: {e}")

    print(f"\n   Всего загружено: {len(documents)} документов")

    if len(documents) == 0:
        print("   ❌ Нет данных в cleaned_dataset/")
        exit(1)

    print("\n2️⃣ Разбивка текста (ОПТИМИЗИРОВАННО)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200  # Больше текста = меньше фрагментов
    )
    splits = text_splitter.split_documents(documents)
    print(f"   ✅ Создано {len(splits)} фрагментов (вместо 1.6 млн!)")

    print("\n3️⃣ Инициализация эмбеддингов...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print("   ✅ Готово")

    print("\n4️⃣ Создание векторной базы (быстро!)...")
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embeddings, persist_directory=str(db_dir)
    )
    print("   ✅ ChromaDB создана")

    print("\n5️⃣ Инициализация LLM...")
    llm = ChatOllama(model="llama3.1:8b", temperature=0)
    print("   ✅ Готова")

    print("\n" + "=" * 70)
    print("✨ ЧАТ-БОТ ГОТОВ ✨")
    print("=" * 70 + "\n")

    while True:
        question = input("❓ Вопрос: ").strip()

        if question.lower() in ["выход", "exit", "quit"]:
            print("👋 До свидания!")
            break

        if not question:
            continue

        print("\n⏳ Анализ...\n")

        try:
            docs = vectorstore.similarity_search(question, k=5)
            context = "\n---\n".join([doc.page_content for doc in docs])

            prompt = ChatPromptTemplate.from_template(
                """Вы — медицинский ассистент.

ДОКУМЕНТЫ:
{context}

ВОПРОС: {question}

ОТВЕТ:"""
            )

            chain = prompt | llm
            response = chain.invoke({"context": context, "question": question})

            print(f"📝 {response.content}\n")

        except Exception as e:
            print(f"❌ Ошибка: {e}\n")

except Exception as e:
    print(f"\n❌ ОШИБКА: {e}\n")
    import traceback

    traceback.print_exc()
