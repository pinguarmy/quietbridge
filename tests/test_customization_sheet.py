from scripts.apply_customization import parse_sheet


def test_single_sheet_separates_profile_words_and_sentences():
    sections = parse_sheet('''# 留注释
[操作习惯]
整理风格：自动整理
[常用名词]
Example Name
# ignored
[常用英文句子]
Can you send the document tomorrow morning?
[试用场景]
Notes
''')
    assert sections['常用名词'] == ['Example Name']
    assert sections['常用英文句子'] == ['Can you send the document tomorrow morning?']
    assert sections['操作习惯'] == ['整理风格：自动整理']
