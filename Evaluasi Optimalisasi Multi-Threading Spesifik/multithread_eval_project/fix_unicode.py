import os
import glob

replacements = {
    '─': '-',
    '→': '->',
    '█': '=',
    '•': '*',
    '✅': '[OK]',
    '⚠️': '[WARN]',
    '🔴': '[ERROR]',
    '×': 'x',
    '−': '-',
    '▸': '>',
    'Δ': 'd',
    '░': '.',
    '■': '*',
}

for filepath in glob.glob('src/*.py'):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    for k, v in replacements.items():
        content = content.replace(k, v)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print("Unicode replacement complete.")
