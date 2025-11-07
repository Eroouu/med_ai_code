import os
import json
from collections import defaultdict

EXPECTED_SECTIONS = {
    "definition",
    "etiology", 
    "epidemiology",
    "clinical_picture",
    "diagnostics",
    "complaints_anamnesis",
    "physical_examination",
    "lab_diagnostics",
    "instrumental_diagnostics",
    "other_diagnostics"
}

def analyze_gaps(data_dir='cleaned_dataset'):
    """Анализирует пропуски в датасете"""
    
    stats = {
        'total_files': 0,
        'missing_sections': defaultdict(int),
        'empty_sections': defaultdict(int),
        'files_with_gaps': []
    }
    
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
    print(f"📊 Анализ {len(files)} файлов...\n")
    
    for fname in files:
        stats['total_files'] += 1
        filepath = os.path.join(data_dir, fname)
        
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
        
        sections = data.get('sections', {})
        title = data.get('title', fname)
        
        present = set(sections.keys())
        missing = EXPECTED_SECTIONS - present
        empty = {k for k, v in sections.items() if not v or not str(v).strip()}
        
        if missing or empty:
            stats['files_with_gaps'].append({
                'file': fname,
                'title': title,
                'missing': missing,
                'empty': empty
            })
        
        for section in missing:
            stats['missing_sections'][section] += 1
        
        for section in empty:
            stats['empty_sections'][section] += 1
    
    return stats

def print_report(stats):
    """Красивый вывод отчета"""
    
    print("=" * 80)
    print("📈 ОТЧЕТ О ПРОПУСКАХ В ДАТАСЕТЕ")
    print("=" * 80)
    
    print(f"\n📁 Всего файлов: {stats['total_files']}")
    print(f"⚠️  Файлов с пропусками: {len(stats['files_with_gaps'])}")
    
    # Статистика по отсутствующим секциям
    if stats['missing_sections']:
        print("\n" + "=" * 80)
        print("❌ ОТСУТСТВУЮЩИЕ СЕКЦИИ (по количеству файлов):")
        print("=" * 80)
        
        for section, count in sorted(stats['missing_sections'].items(), 
                                     key=lambda x: -x[1]):
            percentage = (count / stats['total_files']) * 100
            print(f"  {section:30s} | {count:4d} файлов ({percentage:5.1f}%)")
    
    # Статистика по пустым секциям
    if stats['empty_sections']:
        print("\n" + "=" * 80)
        print("📭 ПУСТЫЕ СЕКЦИИ (по количеству файлов):")
        print("=" * 80)
        
        for section, count in sorted(stats['empty_sections'].items(), 
                                     key=lambda x: -x[1]):
            percentage = (count / stats['total_files']) * 100
            print(f"  {section:30s} | {count:4d} файлов ({percentage:5.1f}%)")
    
    # Детальный список файлов с пропусками
    if stats['files_with_gaps']:
        print("\n" + "=" * 80)
        print("📄 ФАЙЛЫ С ПРОПУСКАМИ (первые 30):")
        print("=" * 80 + "\n")
        
        for i, item in enumerate(stats['files_with_gaps'][:30]):
            print(f"{i+1}. 📝 {item['file']}")
            print(f"   Название: {item['title']}")
            
            if item['missing']:
                print(f"   ❌ Отсутствуют: {', '.join(sorted(item['missing']))}")
            
            if item['empty']:
                print(f"   📭 Пустые: {', '.join(sorted(item['empty']))}")
            
            print()
        
        if len(stats['files_with_gaps']) > 30:
            print(f"   ... и еще {len(stats['files_with_gaps']) - 30} файлов")

if __name__ == "__main__":
    stats = analyze_gaps()
    print_report(stats)
