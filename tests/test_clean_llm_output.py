import pytest
from utils import clean_llm_output


def test_clean_fenced_json_with_language():
    s = '```json\n[{"a":1}]\n```'
    assert clean_llm_output(s, language='json') == '[{"a":1}]'


def test_clean_fenced_json_no_language():
    s = 'Some text\n```\n[1,2,3]\n```\nmore'
    assert clean_llm_output(s, language='json') == '[1,2,3]'


def test_clean_fenced_json_with_surrounding_text():
    inner = '{\n  "id": 1,\n  "name": "Alice"\n}'
    s = f'Output below:\n```json\n{inner}\n```\nEnd'
    assert clean_llm_output(s, language='json') == inner.strip()


def test_no_fence_returns_stripped():
    s = '  [ {"id": 1} ]  '
    assert clean_llm_output(s, language='json') == s.strip()
