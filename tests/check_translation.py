"""Validate the compiled Arabic catalog against every literal gettext call."""
import ast
import gettext
from pathlib import Path
from string import Formatter

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'addon/locale/ar/LC_MESSAGES/nvda.mo'
with path.open('rb') as stream:
    catalog = gettext.GNUTranslations(stream)
sources = [ROOT / 'buildVars.py', ROOT / 'addon/installTasks.py']
sources += list((ROOT / 'addon/globalPlugins/offlineAI').glob('*.py'))
messages = set()
for source in sources:
    tree = ast.parse(source.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == '_' and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            messages.add(node.args[0].value)


def fields(text):
    return sorted((name, spec, conversion or '') for _, name, spec, conversion
                  in Formatter().parse(text) if name is not None)


for message in messages:
    assert message in catalog._catalog, f'Missing translation: {message!r}'
    translated = catalog.gettext(message)
    assert translated.strip(), f'Empty translation: {message!r}'
    assert fields(message) == fields(translated), f'Changed placeholders: {message!r}'
    assert message.replace('&&', '').count('&') == translated.replace('&&', '').count('&'), (
        f'Changed access keys: {message!r}')
    if '|' in message:
        assert message.split('|')[1::2] == translated.split('|')[1::2], (
            f'Changed file patterns: {message!r}')
print(f'Validated {len(messages)} Arabic messages, placeholders, access keys and file filters.')
