import aiohttp

from telegram.ext import (
    Application,
    CommandHandler,
    filters,
    MessageHandler,
    CallbackQueryHandler,
)
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from config import ANSWER_URL, FEEDBACK_URL, TG_TOKEN
from loguru import logger


async def start(update, context):
    chat_id = update.effective_chat.id
    try:
        with open("start.jpg", "rb") as photo:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=photo,
                caption="Привет! Я корпоративный бот для помощи операторам технической поддержки RuTube. Для того чтобы начать, напиши мне вопрос пользователя, на который требуется написать ответ.",
            )
    except FileNotFoundError:
        await context.bot.send_message(
            chat_id=chat_id, text="Привет! Я корпоративный бот для помощи операторам технической поддержки RuTube. Для того чтобы начать, напиши мне вопрос пользователя, на который требуется написать ответ."
        )


async def handle_message(update, context):
    chat_id = update.effective_chat.id
    user = update.effective_user

    feedback_message_id = context.user_data.get("feedback_message_id")
    if feedback_message_id:
        try:
            await context.bot.delete_message(
                chat_id=chat_id, message_id=feedback_message_id
            )
        except BadRequest:
            pass
        context.user_data["feedback_message_id"] = None

    question_text = update.message.text or update.message.caption or ""
    photo_file_id = None
    if update.message.photo:
        photo = update.message.photo[-1]
        photo_file_id = photo.file_id

    payload = {"user_id": user.id}

    if question_text:
        payload["text"] = question_text
    if photo_file_id:
        photo_file = await context.bot.get_file(photo_file_id)
        photo_bytes = await photo_file.download_as_bytearray()
        payload["photo"] = photo_bytes

    temp_message = await context.bot.send_message(
        chat_id=chat_id,
        text="Генерируем ответ...",
    )

    async with aiohttp.ClientSession() as session:
        try:
            data = {}
            if question_text:
                data["question"] = question_text
            async with session.post(ANSWER_URL, json=data) as resp:
                if resp.status == 200:
                    answer_response = await resp.json()
                    answer_text = answer_response.get("answer", "")
                    answer_photo_url = None
                else:
                    logger.error(f"Failed to send question to server, resp={await resp.text()}")
                    answer_text = "Произошла ошибка при получении ответа."
                    answer_photo_url = None
        except Exception as e:
            logger.error(f"Error: {e}")
            answer_text = "Ошибка при обращении к серверу."
            answer_photo_url = None

    try:
        await context.bot.delete_message(
            chat_id=chat_id, message_id=temp_message.message_id
        )
    except BadRequest:
        pass

    if answer_photo_url:
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=answer_photo_url,
            caption=answer_text,
        )
    else:
        await context.bot.send_message(
            chat_id=chat_id,
            text=answer_text,
        )

    keyboard = [
        [
            InlineKeyboardButton("👍 Да", callback_data="feedback_yes"),
            InlineKeyboardButton("👎 Нет", callback_data="feedback_no"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    feedback_message = await context.bot.send_message(
        chat_id=chat_id, text="Был ли ответ полезен?", reply_markup=reply_markup
    )

    context.user_data["feedback_message_id"] = feedback_message.message_id

    context.user_data["last_question"] = question_text
    context.user_data["last_answer"] = answer_text


async def feedback_handler(update: Update, context):
    query = update.callback_query
    await query.answer()

    user_response = query.data
    chat_id = update.effective_chat.id

    question = context.user_data.get("last_question", "")
    answer = context.user_data.get("last_answer", "")

    if user_response == "feedback_yes":
        feedback = True
        feedback_text = "Спасибо за вашу обратную связь!"
    else:
        feedback = False
        feedback_text = "Спасибо, мы учтём вашу обратную связь."

    payload = {
        "question": question,
        "answer": answer,
        "satisfaction": feedback,
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(FEEDBACK_URL, json=payload) as resp:
                if resp.status == 200:
                    pass
                else:
                    logger.error(f"Failed to send feedback to server, resp={await resp.text()}")
        except Exception as e:
            logger.error(f"Error sending feedback: {e}")

    feedback_message_id = context.user_data.get("feedback_message_id")
    if feedback_message_id:
        try:
            await context.bot.delete_message(
                chat_id=chat_id, message_id=feedback_message_id
            )
        except BadRequest:
            pass
        context.user_data["feedback_message_id"] = None

    await context.bot.send_message(chat_id=chat_id, text=feedback_text)


def main():
    application = Application.builder().token(TG_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(
        MessageHandler(filters.TEXT | filters.PHOTO, handle_message)
    )
    application.add_handler(
        CallbackQueryHandler(feedback_handler, pattern="^feedback_")
    )

    application.run_polling()


if __name__ == "__main__":
    main()
