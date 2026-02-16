import torch
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from typing import List, Dict, Optional
from datetime import datetime, date
from transformers import GPT2Tokenizer, T5ForConditionalGeneration
import pytz

# Настройки временной зоны
IRKUTSK_TZ = pytz.timezone('Asia/Irkutsk')
UTC_TZ = pytz.UTC
messages_storage: Dict[int, List[Dict]] = {}

# Загрузка модели
device = torch.device('cuda')
model_name = "RussianNLP/FRED-T5-Summarizer"

print(f"Загрузка модели {model_name} на {device}...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name, eos_token='</s>')
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()
print("Модель загружена!")


async def generate_with_prompt(chat_id: int, prompt_prefix: str, max_new_tokens: int = 300, min_new_tokens: int = 50) -> str:
    """
    Универсальная функция для генерации текста моделью на основе заданного промпта.
    prompt_prefix — начало промпта (без диалога), например: "<LM> Перескажи диалог подробно.\n"
    """
    dialog = get_dialog_text(chat_id)
    if not dialog:
        return "Нет сообщений для анализа."

    input_text = prompt_prefix + dialog
    input_ids = torch.tensor([tokenizer.encode(input_text)]).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=5,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            length_penalty=0.6,
            early_stopping=False,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_p=0.92,
            temperature=0.8,
            repetition_penalty=1.1
        )

    result = tokenizer.decode(outputs[0][1:], skip_special_tokens=True)
    return result

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in messages_storage:
        messages_storage[chat_id] = []
    await update.message.reply_text(
        "👋 Привет! Я бот для анализа диалогов.\n"
        "/log — подробный пересказ последних сообщений\n""
        "/clear — очистить историю"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    if chat_id not in messages_storage:
        messages_storage[chat_id] = []
    messages_storage[chat_id].append({
        'name': update.effective_user.first_name,
        'text': update.message.text,
        #'time': format_telegram_date(update.message.date)
    })

async def log_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in messages_storage or not messages_storage[chat_id]:
        await update.message.reply_text("📭 История пуста")
        return

    processing_msg = await update.message.reply_text("⏳ Составляю подробный пересказ...")
    try:
        prompt = "<LM> Перескажи диалог максимально подробно, сохраняя все ключевые моменты, имена участников и их действия. Не упускай детали.\n"
        summary = await generate_with_prompt(chat_id, prompt, max_new_tokens=512, min_new_tokens=64)
        msg_count = len(messages_storage[chat_id])
        await processing_msg.edit_text(
            f"📋 **Подробный анализ диалога:**\n\n{summary}\n\n---\n*Сообщений в истории: {msg_count}*"
        )
    except Exception as e:
        await processing_msg.edit_text(f"❌ Ошибка: {str(e)}")

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in messages_storage:
        messages_storage[chat_id] = []
        await update.message.reply_text("🧹 История очищена")
    else:
        await update.message.reply_text("📭 История и так пуста")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)  # просто вызываем start

def start_bot():
    application = Application.builder().token("BOT-TOKEN").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("log", log_command))
    #application.add_handler(CommandHandler("topics", topics_command))
    #application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("clear", clear_command))
    #application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен...")
    application.run_polling()

def get_dialog_text(chat_id: int) -> Optional[str]:
    """
    Возвращает текст диалога (имя: сообщение) для указанного чата.
    """
    if chat_id not in messages_storage or not messages_storage[chat_id]:
        return None

    msgs = messages_storage[chat_id]

    if not msgs:
        return None

    dialog = "\n".join([f"{m['name']}: {m['text']}" for m in msgs])
    return dialog


start_bot()