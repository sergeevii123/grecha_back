from functools import partial

import pandas as pd

issues_columns_rename_dict = {
    'ИД выдачи': 'issue_id',
    'ИД читателя': 'reader_id',
    'Дата выдачи': 'issue_date',
    'Инвентарный номер': 'inventory_id',
    'Штрих-код': 'barcode',
    'Дата сдачи (предполагаемая)': 'return_date',
    'Состояние': 'condition'
}

examples_columns_rename_dict = {
    'Идентификатор экземпляра': 'example_id',
    'ИД Каталожной записи': 'record_id',
    'Инвентарный номер': 'inventory_id',
    'Штрих-код': 'barcode',
    'Раздел знаний': 'knowledge_id',
    'Идентификатор сиглы': 'sigly_id',
}

readers_rename_dict = {
    'ID читателя': 'reader_id',
    'Дата рождения': 'birth_date'
}

catalogue_rename_dict = {
    'doc_id': 'record_id',
    'p100a': 'author',
    'p245a': 'title',
    'p260a': 'city',
    'p260b': 'publisher',
    'p260c': 'year',
    'p490a': 'series',
    'p650a': 'genres',
    'p084a': 'knowledge_id',
    'p521a': 'age_rating',


}


def prepare_df_from_excel(path: str, rename_dict: dict) -> pd.DataFrame:
    df = pd.ExcelFile(path)
    sheets = {}
    for sheet in df.sheet_names:
        sheets[sheet] = df.parse(sheet)

    columns_to_use = list(rename_dict.values())
    return pd.concat([
        df.rename(columns=rename_dict)[columns_to_use]
        for df in sheets.values()
    ])


prepare_issues_df = partial(prepare_df_from_excel, rename_dict=issues_columns_rename_dict)
prepare_examples_df = partial(prepare_df_from_excel, rename_dict=examples_columns_rename_dict)
prepare_readers_df = partial(prepare_df_from_excel, rename_dict=readers_rename_dict)
prepare_catalogue_df = partial(prepare_df_from_excel, rename_dict=catalogue_rename_dict)


if __name__ == '__main__':
    issues_1 = prepare_issues_df('data/Выдача_1.xlsx')
    issues_2 = prepare_issues_df('data/Выдача_2.xlsx')
    issues_full = pd.concat([issues_1, issues_2])
    issues_full.to_csv('data/issues.csv', index=False)

    examples_1 = prepare_examples_df('data/Экземпляры.xlsx')
    examples_2 = prepare_examples_df('data/Экземпляры_2.xlsx')
    examples_full = pd.concat([examples_1, examples_2])
    examples_full.to_csv('data/examples.csv', index=False)

    readers = prepare_readers_df('data/Читатели.xlsx')
    readers.to_csv('data/readers.csv', index=False)

    catalogue = prepare_catalogue_df('data/Каталог.xlsx')
    catalogue.to_csv('data/catalogue.csv', index=False)

