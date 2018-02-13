FROM python:2
ADD . /usr/src/app
WORKDIR /usr/src/app
RUN pip install -r requirements.txt
EXPOSE 80
EXPOSE 8080
EXPOSE 6005

# Define environment variable
ENV NAME World
CMD ["python", "app.py"]