# Зависимости
python 3.7+

``pip3 install -r requirements.txt``

# Подготовка сырых данных 
В папке data должны лежать все xlsx файлы [отсюда](https://drive.google.com/drive/folders/1AQjt73rLgM7EBapqGjifzcMFOmiH0-MF).
```
python3 prepare_data.py
```

# Подготовка моделей для книг
```
python3 prepare_library.py
python3 train_model.py
python3 train_authors_model.py
python3 train_user_history_py.py
python3 book_sampler.py
```

# Подготовка моделей для КДФ
В папке data должны лежать все файлы [отсюда](https://drive.google.com/drive/folders/13Egz5APU8eJZ8N8md0Pm2xb7E3d7cJxK).
```
python3 train_kdf.py
python3 prepare_kdf_rec.py
pyhton3 prepare_reestr.py

```

# Запуск приложения
``python3 server.py``

# Фронт 
ios приложение делалось в отдельном репозитории

линк к нему в папке front
