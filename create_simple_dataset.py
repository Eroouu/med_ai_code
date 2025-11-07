import os
import json
from bs4 import BeautifulSoup
import re

# ТОЛЬКО ЭТИ СЕКЦИИ (без additional_info!)
ALLOWED_SECTIONS = {
    "definition": ["Определение заболевания", "Определение"],
    "etiology": ["Этиология"],
    "epidemiology": ["Эпидемиология"],
    "clinical_picture": ["Клиническая картина"],
    "diagnostics": ["Диагностика", "Критерии установления диагноза"],
    "complaints_anamnesis": ["Жалобы и анамнез", "Рекомендуется при сборе жалоб"],
    "physical_examination": ["Физикальное обследование", "Рекомендуется пациентам при подозрении"],
    "lab_diagnostics": ["Лабораторные диагностические исследования", "Лабораторная диагностика"],
    "instrumental_diagnostics": ["Инструментальные диагностические исследования", "Инструментальная диагностика"],
    "other_diagnostics": ["Иные диагностические исследования", "Диагностические ангиографические методики"]
}

def clean_text(text):
    """Убираем лишние символы и пробелы"""
    text = re.sub(r'\n+', '\n', text)  # Множественные переносы в один
    text = re.sub(r' +', ' ', text)     # Множественные пробелы в один
    return text.strip()

def parse_html_clean(filepath):
    """Парсинг ТОЛЬКО разрешенных секций"""
    with open(filepath, encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Извлекаем title
    title = soup.title.get_text(strip=True) if soup.title else os.path.basename(filepath)

    # Получаем весь текст
    body_text = soup.body.get_text(separator='\n', strip=True)
    lines = [line.strip() for line in body_text.split('\n') if line.strip()]

    # Парсим секции по номерам (например "1.2 Этиология")
    section_pattern = re.compile(r"^([0-9]+(?:\.[0-9]+)*\.?\s+.+)")
    sections_raw = {}
    current_section = None
    buffer = []
    
    for line in lines:
        if section_pattern.match(line):
            if current_section:
                sections_raw[current_section] = "\n".join(buffer).strip()
                buffer = []
            current_section = line
        else:
            if current_section:
                buffer.append(line)
                
    if current_section and buffer:
        sections_raw[current_section] = "\n".join(buffer).strip()

    # Фильтруем ТОЛЬКО нужные секции
    final_sections = {}
    
    for key, patterns in ALLOWED_SECTIONS.items():
        for section_name, section_text in sections_raw.items():
            # Проверяем совпадение с любым из паттернов
            if any(pattern.lower() in section_name.lower() for pattern in patterns):
                final_sections[key] = clean_text(section_text)
                break  # Берем первое совпадение

    # Создаем результат БЕЗ additional_info и images
    output = {
        "title": title,
        "sections": final_sections
    }

    # Сохраняем
    basename = os.path.splitext(os.path.basename(filepath))[0]
    outdir = os.path.join(os.getcwd(), 'cleaned_dataset')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'{basename}.json')
    
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✅ {basename}: {len(final_sections)} секций")
    return outpath

def batch_parse(directory='clin_rec_html'):
    """Обрабатываем все HTML файлы"""
    if not os.path.exists(directory):
        print(f"❌ Папка {directory} не найдена!")
        return
    
    files = [f for f in os.listdir(directory) if f.lower().endswith('.html')]
    print(f"📁 Найдено {len(files)} HTML файлов\n")
    
    success = 0
    for filename in files:
        filepath = os.path.join(directory, filename)
        try:
            parse_html_clean(filepath)
            success += 1
        except Exception as e:
            print(f"❌ Ошибка в {filename}: {e}")
    
    print(f"\n✨ Готово! Обработано {success}/{len(files)} файлов")
    print(f"📂 Датасет сохранен в 'cleaned_dataset/'")

if __name__ == "__main__":
    batch_parse()
