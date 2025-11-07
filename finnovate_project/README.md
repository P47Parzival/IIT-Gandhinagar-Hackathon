# IIT-Gn Finnovate 2025 Hackathon

## For Local:-

### 1. Place .env parallel to manage.py

### 2. From root (finnovate_project), run
```
python -m venv env
env/scripts/activate
pip install -r requirements.txt
```

### 3. Run migrations
```
python fintech_project/manage.py makemigrations
```

### 4. Migrate
```
python fintech_project/manage.py migrate
```

### 5. Create staticfiles directory: just run the command
```
python fintech_project/manage.py collectstatic
```

### 6. Install NGINX (SKIP THIS ACTUALLY !!)
```
choco install nginx
```
If you have issues, check this
"C:\ProgramData\chocolatey\lib\nginx\tools\"
There might be a zip named "nginx-1.29.3.zip" instead of a DIR
Extract it right there
Then add "C:\ProgramData\chocolatey\lib\nginx\tools\nginx-1.29.3" to System Variables PATH
Test with `nginx -v`

### 6. Create superuser
```
python fintech_project/manage.py createsuperuser
```

### 7. To run from finnovate_project>
```
python fintech_project/manage.py runserver
```

### 8. Login using admin creds
Th ones you used during createsuperuser cmd

## FOR PROD:-
Close the server if it is running on Windows terminal, we'll use NGINX+Gunicorn in WSL
### 9. (in WSL) 
```
cd /mnt/c/Users/krish/Desktop/FINTECH/finnovate_project
sudo apt install python-is-python3 -y
pip install -r requirements.txt
```
You'll need to restart WSL
```
cd /mnt/c/Users/krish/Desktop/FINTECH/finnovate_project/fintech_project
gunicorn --bind 0.0.0.0:8000 core.wsgi
```
Update the paths in your nginx.conf and test it by opening a new WSL terminal and running
```
sudo apt install nginx-core
sudo nginx -t -c /mnt/c/Users/krish/Desktop/FINTECH/finnovate_project/deploy/nginx/nginx.conf
```
Test it using
```
sudo nginx -t -c /mnt/c/Users/krish/Desktop/FINTECH/finnovate_project/deploy/nginx/nginx.conf
```
On one side, you need gunicorn to keep running from fintech_project using 
```
gunicorn --bind 0.0.0.0:8000 core.wsgi
```
On other side, you need NGINX to keep running using
```
sudo nginx -c /mnt/c/Users/krish/Desktop/FINTECH/finnovate_project/deploy/nginx/nginx.conf
```
Test using
```
curl localhost:8081
```
To stop it, use
```
sudo nginx -s stop
```

Also, I kinda shouldn't need to tell you this if you actually read this file
But we'll be using localhost:8081 now, not port 8000
Gunicorn listens to 8000
NGINX listens to 8081
Please dont bypass NGINX, it is out cute little reverse proxy setup

## SAP ERP (HANA 2.0)

### Important Links:-
https://www.sap.com/products/data-cloud/hana/express-trial.html
https://tools.hana.ondemand.com/#hanatools

https://account.hanatrial.ondemand.com/trial/#/home/trial

TEST SAP HANA ERP CONNECTION:-

```
python -c "from hdbcli import dbapi; conn = dbapi.connect(address='ebdc3fa3-fb21-454f-bad4-f569d264fd7c.hana.trial-us10.hanacloud.ondemand.com', port=443, user='DBADMIN', password='AdaniPower@123', encrypt=True, sslValidateCertificate=False); print('Connected!'); conn.close()"
```