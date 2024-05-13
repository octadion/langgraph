from langchain_community.utilities import SQLDatabase

url = f"postgresql+psycopg2://postgres:admin@localhost:5432/anggota"

db = SQLDatabase.from_uri(
    url,
    schema="komunitas_anggota",
    include_tables=['anggota'],
    sample_rows_in_table_info=1,
)

def get_schema(_):
    schema = db.get_table_info()
    return schema