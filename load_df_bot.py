import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
import telebot

from token_file import TOKEN

bot = telebot.TeleBot(TOKEN, parse_mode=None)
df = None

def preprocess_data(message):
    df_temp = df.copy()
    Y_train = df_temp.pop(message.text)
    X_train = df_temp.copy()
    X_train_num, X_train_cat = X_train.select_dtypes(include=np.number), X_train.select_dtypes(exclude=np.number)
    X_train_cat = OneHotEncoder(sparse=False).fit_transform(X_train_cat)
    X_train = np.concatenate([X_train_num, X_train_cat], axis=-1)
    return X_train, Y_train

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, '馃 Saludos, humano! 馃\nUtiliza /help para ver todos los comandos disponibles.')

@bot.message_handler(commands=['help', 'ayuda'])
def send_welcome(message):
    msg = """隆Bienvenid@! 馃憢

Este bot est谩 preparado para realizar una presentaci贸n de los servicios de AutoML de nuestra plataforma. 
Antes de poder utilizarlo tendr谩s que subir un archivo de datos en formato .csv.
Los comandos disponibles son:

*Visualizaci贸n* 馃搳
/histograma -> Realiza un histograma de la variable que le pidas.
/boxplot -> Realiza un boxplot de la variable que le pidas.

*Predicci贸n* 馃敭
/clasificacion -> Ajusta una regresi贸n log铆stica para predecir la variable deseada.
/regresion -> Ajusta un SVM para predecir la variable deseada.

*Misc.* 馃
/contacto -> Muestra nuestra p谩gina web para que puedas obtener m谩s informaci贸n de nuestro servicio de pago.
    """
    bot.reply_to(message, msg, parse_mode='Markdown')

@bot.message_handler(content_types=['document'])
def download_df(message):
    file_id = message.document.file_id
    file_info = bot.get_file(file_id)
    file = requests.get(f'https://api.telegram.org/file/bot{TOKEN}/{file_info.file_path}')
    open('test.csv', 'wb').write(file.content)
    global df 
    df = pd.read_csv('test.csv')
    bot.reply_to(message, 'El DataFrame tiene {} filas y {} columnas.'.format(*df.shape))

@bot.message_handler(commands=['histograma'])
def start_histogram(message):
    if df is None:
        bot.reply_to(message, 'Primero tienes que subir un archivos de datos. 馃槧')
        return
    bot.set_state(user_id=message.from_user.id, 
                  state='hist', 
                  chat_id=message.chat.id)
    markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True, 
                                               selective=False)
    items = [telebot.types.KeyboardButton(var) for var in df.columns]
    markup.add(*items)
    bot.send_message(message.chat.id, 'Elige una variable.\n'+
                                      'En la versi贸n de prueba solamente est谩 permitido visualizar una variable.', 
                     reply_markup=markup)

@bot.message_handler(func=lambda m: bot.get_state(m.from_user.id, m.chat.id)=='hist' and m.text in df.columns)
def plot_histogram(message):
    plt.figure()
    df.hist(message.text)
    plt.savefig('/tmp/photo.png')
    with open('/tmp/photo.png', 'rb') as photo:
        bot.send_photo(message.chat.id, photo)
    bot.set_state(user_id=message.from_user.id, 
                  state=None, 
                  chat_id=message.chat.id)

@bot.message_handler(commands=['boxplot'])
def start_boxplot(message):
    if df is None:
        bot.reply_to(message, 'Primero tienes que subir un archivos de datos. 馃槧')
        return
    bot.set_state(user_id=message.from_user.id, 
                  state='boxplot', 
                  chat_id=message.chat.id)
    markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True, 
                                               selective=False)
    items = [telebot.types.KeyboardButton(var) for var in df.columns]
    markup.add(*items)
    bot.send_message(message.chat.id, 'Elige una variable.\n'+
                                      'En la versi贸n de prueba solamente est谩 permitido visualizar una variable.', 
                     reply_markup=markup)

@bot.message_handler(func=lambda m: bot.get_state(m.from_user.id, m.chat.id)=='boxplot' and m.text in df.columns)
def plot_boxplot(message):
    plt.figure()
    df.boxplot(message.text)
    plt.savefig('/tmp/photo.png')
    with open('/tmp/photo.png', 'rb') as photo:
        bot.send_photo(message.chat.id, photo)
    bot.set_state(user_id=message.from_user.id, 
                  state=None, 
                  chat_id=message.chat.id)

@bot.message_handler(commands=['clasificacion'])
def start_classification_model(message):
    if df is None:
        bot.reply_to(message, 'Primero tienes que subir un archivos de datos. 馃槧')
        return
    bot.set_state(user_id=message.from_user.id, 
                  state='clasificacion', 
                  chat_id=message.chat.id)
    markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True, 
                                               selective=False)
    items = [telebot.types.KeyboardButton(var) for var in df.columns]
    markup.add(*items)
    bot.send_message(message.chat.id, 
                     'Indica qu茅 variable quieres predecir.\n'+ 
                     'En la versi贸n de prueba solamente est谩 permitido predecir una variable.', reply_markup=markup)

@bot.message_handler(func=lambda m: bot.get_state(m.from_user.id, m.chat.id)=='clasificacion' and m.text in df.columns)
def train_classification_model(message):
    X_train, Y_train = preprocess_data(message)
    results = cross_val_score(LogisticRegression(), X_train, Y_train, cv=5)

    bot.reply_to(message, f'La regresi贸n log铆stica obtiene una precisi贸n de {results.mean():.2f} +- {results.std():.2f}.\n'+
                           'Recuerda que si quieres poder guardar el modelo o ajustar los par谩metros necesitas una suscripci贸n de pago. 馃')
    
    bot.set_state(user_id=message.from_user.id, 
                  state=None, 
                  chat_id=message.chat.id)

@bot.message_handler(commands=['regresion'])
def start_classification_model(message):
    if df is None:
        bot.reply_to(message, 'Primero tienes que subir un archivos de datos. 馃槧')
        return
    bot.set_state(user_id=message.from_user.id, 
                  state='regresion', 
                  chat_id=message.chat.id)
    markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True, 
                                               selective=False)
    items = [telebot.types.KeyboardButton(var) for var in df.columns]
    markup.add(*items)
    bot.send_message(message.chat.id, 
                     'Indica qu茅 variable quieres predecir.\n'+ 
                     'En la versi贸n de prueba solamente est谩 permitido predecir una variable.', reply_markup=markup)

@bot.message_handler(func=lambda m: bot.get_state(m.from_user.id, m.chat.id)=='regresion' and m.text in df.columns)
def train_classification_model(message):
    X_train, Y_train = preprocess_data(message)
    results = cross_val_score(SVR(), X_train, Y_train, cv=5)

    bot.reply_to(message, f'El SVM obtiene un coeficiente de correlaci贸n R2 de {results.mean():.2f} +- {results.std():.2f}.\n'+
                           'Recuerda que si quieres poder guardar el modelo o ajustar los par谩metros necesitas una suscripci贸n de pago. 馃')
    
    bot.set_state(user_id=message.from_user.id, 
                  state=None, 
                  chat_id=message.chat.id)

@bot.message_handler(commands=['contacto'])
def send_contact(message):
    bot.reply_to(message, 'Puedes encontrar m谩s informaci贸n sobre nuestros servicios en: https://thisstartupdoesnotexist.com/')

## Starting the bot
bot.infinity_polling()