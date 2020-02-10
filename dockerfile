FROM python:3.8

RUN pip install torch torchvision

RUN pip install streamlit

RUN pip install spacy
RUN pip install -U spacy-lookups-data
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download es_core_news_md