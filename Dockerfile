FROM inemo/isanlp_base

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y liblzma-dev

COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt && pip install grpcio git+https://github.com/IINemo/isanlp.git

COPY src/isanlp_clause_segmenter /src/isanlp_clause_segmenter
COPY pipeline_object.py /src/isanlp_clause_segmenter/pipeline_object.py

COPY models /models

ENV PYTHONPATH=/src/isanlp_clause_segmenter/
CMD ["python", "/start.py", "-m", "pipeline_object", "-a", "create_pipeline", "--no_multiprocessing", "True"]
