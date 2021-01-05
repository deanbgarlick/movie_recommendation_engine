FROM tensorflow/tensorflow

WORKDIR /home/app

RUN pip install -q tensorflow-recommenders
RUN pip install -q --upgrade tensorflow-datasets
RUN pip install pandas numpy jupyter matplotlib seaborn sklearn corextopic

CMD ['jupyter', 'notebook', '--ip=0.0.0.0', '--port=8888', '--allow-root']