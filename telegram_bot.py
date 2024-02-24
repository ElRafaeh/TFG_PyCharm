# t.me/Rafa_Bustos_bot
import threading
import io

import telebot
import cv2

chat_id = 332779048
BOT_TOKEN = '6838227610:AAEosRwmkr8q2iKtZGVXswB9A-H4spWiwsc'
bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "Bienvenido a mi bot detecta caidas")

def send_photo_example():
    img = cv2.imread('images/mayor.png')

    ret, img_encode = cv2.imencode('.jpg', img)
    str_encode = img_encode.tobytes()
    img_byteio = io.BytesIO(str_encode)
    img_byteio.name = 'img.jpg'
    reader = io.BufferedReader(img_byteio)
    bot.send_photo(chat_id, reader)


threading.Thread(target=send_photo_example).start()
bot.infinity_polling()