import requests
import pandas as pd
import matplotlib.pyplot as plt
import telebot

from token_file import TOKEN

bot = telebot.TeleBot(TOKEN, parse_mode=None)
df = None

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, 'Saludos, humano!')

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
def plot_histogram(message):
    global state
    state = 'hist'
    bot.set_state(user_id=message.from_user.id, 
                  state='hist', 
                  chat_id=message.chat.id)
    markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True, 
                                               selective=False)
    items = [telebot.types.KeyboardButton(var) for var in df.columns]
    markup.add(*items)
    bot.send_message(message.chat.id, 'Elige una variable:', reply_markup=markup)

@bot.message_handler(func=lambda m: m.text in df.columns and bot.get_state(m.from_user.id, m.chat.id)=='hist')
def plot_histogram(message):
    df.hist(message.text)
    plt.savefig('/tmp/photo.png')
    with open('/tmp/photo.png', 'rb') as photo:
        bot.send_photo(message.chat.id, photo)
    bot.set_state(user_id=message.from_user.id, 
                  state=None, 
                  chat_id=message.chat.id)

## Starting the bot
bot.infinity_polling()