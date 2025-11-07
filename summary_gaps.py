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

def get_gaps_summary(data_dir='cleaned_dataset'):
    """Получает итоговую статистику пропусков"""
    
    gap_stats = {section: {'отсутствует': 0, 'пусто': 0} for section in EXPECTED_SECTIONS}
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    total_files = len(files)
    
    print(f"Анализируем {total_files} файлов...\n")
    
    for fname in files:
        filepath = os.path.join(data_dir, fname)
        
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
        
        sections = data.get('sections', {})
        present = set(sections.keys())
        
        # Отсутствующие секции
        missing = EXPECTED_SECTIONS - present
        for section in missing:
            gap_stats[section]['отсутствует'] += 1
        
        # Пустые секции
        for section in present:
            if not sections[section] or not str(sections[section]).strip():
                gap_stats[section]['пусто'] += 1
    
    return gap_stats, total_files

def print_summary(gap_stats, total_files):
    """Красиво выводит сводку пропусков"""
    
    print("\n" + "=" * 90)
    print(" " * 25 + "ИТОГОВАЯ СТАТИСТИКА ПРОПУСКОВ")
    print("=" * 90)
    print(f"\nВсего файлов в датасете: {total_files}\n")
    
    print(f"{'Секция':<30} {'Отсутствует':<15} {'Пусто':<15} {'Всего пропусков':<15}")
    print("-" * 90)
    
    total_gaps = 0
    for section in sorted(EXPECTED_SECTIONS):
        missing = gap_stats[section]['отсутствует']
        empty = gap_stats[section]['пусто']
        total = missing + empty
        total_gaps += total
        
        missing_pct = (missing / total_files * 100) if total_files > 0 else 0
        empty_pct = (empty / total_files * 100) if total_files > 0 else 0
        total_pct = (total / total_files * 100) if total_files > 0 else 0
        
        print(f"{section:<30} {missing:>3} ({missing_pct:>5.1f}%)    {empty:>3} ({empty_pct:>5.1f}%)    {total:>3} ({total_pct:>5.1f}%)")
    
    print("-" * 90)
    print(f"{'ИТОГО ПРОПУСКОВ':<30} {'':<15} {'':<15} {total_gaps}")
    print("=" * 90)
    
    # Средний процент полноты
    avg_completeness = ((total_files * len(EXPECTED_SECTIONS) - total_gaps) / (total_files * len(EXPECTED_SECTIONS)) * 100) if total_files > 0 else 0
    print(f"\n✅ Средняя полнота датасета: {avg_completeness:.1f}%")
    print(f"❌ Средний процент пропусков: {100 - avg_completeness:.1f}%\n")

def save_summary_to_file(gap_stats, total_files, output_file='gap_summary.txt'):
    """Сохраняет сводку в текстовый файл"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write(" " * 25 + "ИТОГОВАЯ СТАТИСТИКА ПРОПУСКОВ\n")
        f.write("=" * 90 + "\n")
        f.write(f"\nВсего файлов в датасете: {total_files}\n\n")
        
        f.write(f"{'Секция':<30} {'Отсутствует':<15} {'Пусто':<15} {'Всего пропусков':<15}\n")
        f.write("-" * 90 + "\n")
        
        total_gaps = 0
        for section in sorted(EXPECTED_SECTIONS):
            missing = gap_stats[section]['отсутствует']
            empty = gap_stats[section]['пусто']
            total = missing + empty
            total_gaps += total
            
            missing_pct = (missing / total_files * 100) if total_files > 0 else 0
            empty_pct = (empty / total_files * 100) if total_files > 0 else 0
            total_pct = (total / total_files * 100) if total_files > 0 else 0
            
            f.write(f"{section:<30} {missing:>3} ({missing_pct:>5.1f}%)    {empty:>3} ({empty_pct:>5.1f}%)    {total:>3} ({total_pct:>5.1f}%)\n")
        
        f.write("-" * 90 + "\n")
        f.write(f"{'ИТОГО ПРОПУСКОВ':<30} {'':<15} {'':<15} {total_gaps}\n")
        f.write("=" * 90 + "\n")
        
        avg_completeness = ((total_files * len(EXPECTED_SECTIONS) - total_gaps) / (total_files * len(EXPECTED_SECTIONS)) * 100) if total_files > 0 else 0
        f.write(f"\n✅ Средняя полнота датасета: {avg_completeness:.1f}%\n")
        f.write(f"❌ Средний процент пропусков: {100 - avg_completeness:.1f}%\n")
    
    print(f"📄 Результат сохранен в {output_file}")

if __name__ == "__main__":
    gap_stats, total_files = get_gaps_summary()
    print_summary(gap_stats, total_files)
    save_summary_to_file(gap_stats, total_files)
