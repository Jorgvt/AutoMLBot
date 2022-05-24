import requests
import pandas as pd
import telebot

from token_file import TOKEN

bot = telebot.TeleBot(TOKEN, parse_mode=None)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, 'Saludos, humano!')

@bot.message_handler(content_types=['document'])
def download_df(message):
    file_id = message.document.file_id
    file_info = bot.get_file(file_id)
    file = requests.get(f'https://api.telegram.org/file/bot{TOKEN}/{file_info.file_path}')
    open('test.csv', 'wb').write(file.content)
    df = pd.read_csv('test.csv')
    bot.reply_to(message, 'El DataFrame tiene {} filas y {} columnas.'.format(*df.shape))

## Starting the bot
bot.infinity_polling()