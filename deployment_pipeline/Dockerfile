# STEP 1: Install base image. Optimized for Python.
FROM python:3.7-slim-buster

# Step 2: Add requirements.txt file 
COPY requirements.txt /requirements.txt

# Step 3:  Install required pyhton dependencies from requirements file
RUN pip install -r requirements.txt

# Step 4: Copy source code in the current directory to the container
ADD . /inclinet

# Step 5: Set working directory to previously added app directory
WORKDIR /inclinet

# Step 6: Expose the port Flask is running on
EXPOSE 3030

# Step 9: Run Flask
#RUN python3 server.py
#CMD ["/bin/sh"]


