# python version 3.12.1 (at least 3.10+ required)

# ============== PREREQUISITES ==============

# Ollama installed and "llm_model" specified in config pulled
# Download ollama at https://www.ollama.com/

# pip install langchain
# pip install langchain_community
# pip install -U sentence-transformers
# pip install chardet
# pip install chromadb

# pip install discord.py

# ===========================================

import os, json, asyncio, time
from datetime import datetime
import functools, typing

def to_thread(func: typing.Callable) -> typing.Coroutine:
	@functools.wraps(func)
	async def wrapper(*args, **kwargs):
		return await asyncio.to_thread(func, *args, **kwargs)
	return wrapper

# LLM model setting ========================================================================================================

from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.docstore.document import Document

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_community.llms import Ollama

from langchain.schema.runnable.config import RunnableConfig
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate, ChatPromptTemplate


class RAGModel:
	class CustomHandler(AsyncCallbackHandler):
		callback: typing.Callable[[str], typing.Awaitable[None]]
		
		def __init__(self):
			self.callback = None
			
		async def on_llm_new_token(self, token: str, **kwargs) -> None:
			if self.callback != None:
				await self.callback(token)
			print(token, end='', flush=True)
			
	text_splitter: RecursiveCharacterTextSplitter
	vectordb: Chroma
	handler: CustomHandler
	config: RunnableConfig
	qa: RetrievalQA
	
	db_path: str
	doc_id_file: str
	
	dirty_flag: bool
	document_ids: dict
	
	# chat_history: list  # currently not used
	
	def __init__(
		self,
		db_path: str,
		doc_id_file: str,
		embed_model: str,
		ollama_model: str,
		context_similarity_threshold: float,
		max_context_count: int
	):
		self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
		self.vectordb = Chroma(embedding_function=HuggingFaceEmbeddings(model_name=embed_model, model_kwargs={"device": "cpu"}), persist_directory=db_path)
		
		self.handler = self.CustomHandler()
		# callback_manager = CallbackManager([self.handler])
		self.config = RunnableConfig(callbacks=[self.handler])
		
		# load prompt template
		with open("prompt_template", "r", encoding="utf8") as f:
			prompt = PromptTemplate.from_template(f.read())
			
		self.qa = RetrievalQA.from_chain_type(
			# llm = Ollama(model=ollama_model, callbacks=callback_manager),
			llm = Ollama(model=ollama_model),
			chain_type = "stuff",  # directly send similar context to prompt
			retriever = self.vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": context_similarity_threshold, "k": max_context_count}),
			return_source_documents = True,  # for reference
			verbose = True,  # to show formatted prompt message on query
			chain_type_kwargs = {"verbose": True, "prompt": prompt}
		)
		
		self.db_path = db_path
		self.doc_id_file = doc_id_file
		
		self.dirty_flag = False
		with open(doc_id_file, "r", encoding="utf8") as f:
			self.document_ids = json.loads(f.read())
	
	def add_document(self, text: str, source: str) -> bool:
		self.remove_document(source)  # avoid duplicate data
		if len(text) == 0:
			return False
		
		doc = Document(page_content=text, metadata={"source": source})
		all_splits = text_splitter.split_documents([doc])
		self.document_ids[source] = self.vectordb.add_documents(documents=all_splits, persist_directory=self.db_path)
		self.dirty_flag = True
		return True

	def remove_document(self, source: str) -> bool:
		if source not in self.document_ids:
			return False
		
		self.vectordb.delete(self.document_ids[source])
		del self.document_ids[source]
		self.dirty_flag = True
		return True
		
	def save_documents(self):
		if self.dirty_flag:
			with open(self.doc_id_file, "w", encoding="utf8") as f:
				f.write(json.dumps(self.document_ids, indent=4))
				
			print(f"\n[{datetime.now()}] Document IDs Saved! ==================\n")
			self.dirty_flag = False
			
	@to_thread
	def auto_save(self):  # non-blocking save function
		self.save_documents()

	async def generate_answer(self, question: str, on_token: typing.Callable[[str], typing.Awaitable[None]]):
		# regist callback on token generated
		self.handler.callback = on_token
		
		# if len(self.chat_history) == 0:
			# result = self.qa({"query": question})
		# else:
			# result = self.qa({"query": HISTORY_QUERY.format('\n'.join(self.chat_history), question)})
			
		result = await self.qa.ainvoke({"query": question}, config=self.config)
			
		answer = result["result"]
		print("")
		if len(result["source_documents"]) > 0:
			answer += "\n\n> ### 參考資料："
			for doc in result["source_documents"]:
				answer += "\n> - https://discord.com/channels/" + doc.metadata["source"]
		
		# 不知道為什麼 AI 每次都把 context 和對話紀錄搞混，先不要加上對話紀錄的功能。想要追問的話就把新舊問題統整成一個直接問他
		# self.chat_history.append("Human: " + question + "\nAssistant: " + result["result"])
		# if len(self.chat_history) > 50:
			# del self.chat_history[0]
		
		return answer

# LLM model setting ========================================================================================================

# load configs
ALL_CONFIG = dict()
with open("config", "r", encoding="utf8") as f:
	configs = f.read().split('\n')
	for config in configs:
		sep = config.find('=')
		if sep > 0:
			ALL_CONFIG[config[:sep].strip()] = config[(sep+1):].strip()


rag_model = RAGModel(
	ALL_CONFIG.get("db_path", "db"),
	ALL_CONFIG.get("doc_id_file", "doc_id.json"),
	ALL_CONFIG.get("embed_model", "intfloat/multilingual-e5-large"),
	ALL_CONFIG.get("ollama_model", "llama3"),
	float(ALL_CONFIG.get("context_similarity_threshold", 0.7)),
	int(ALL_CONFIG.get("max_context_count", 4))
)


# set saving event on exit
if os.name == "nt":  # is windows
	try:
		import win32api
	except:
		os.system("pip3 install pywin32")
		import win32api
	import win32con
	
	def handler(ctrlType):
		rag_model.save_documents()
		if ctrlType == win32con.CTRL_C_EVENT:
			raise KeyboardInterrupt()
		return True
	win32api.SetConsoleCtrlHandler(handler, True)
elif os.name == "posix":  # is linux
	import signal
	def handler(*args):
		rag_model.save_documents()
		exit()
	signal.signal(signal.SIGTERM, handler)
	signal.signal(signal.SIGINT, handler)


# discord bot ======================================================================================================

import discord
from discord import app_commands


BOOKMARK_EMOJI = "📑"
DISCORD_SERVER_ID = int(ALL_CONFIG["discord_server_id"])
AUTO_RECORD_PIN_MESSAGE = eval(ALL_CONFIG.get("auto_record_pin_message", True))

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

self_member = None
is_responding = False

async def auto_save_loop():
	while True:
		await asyncio.sleep(300)  # auto save every 5 minutes
		await rag_model.auto_save()

@client.event
async def on_ready():
	global self_member
	await tree.sync(guild=discord.Object(id=DISCORD_SERVER_ID))
	
	guild = client.get_guild(DISCORD_SERVER_ID)
	self_member = guild.get_member(client.user.id)
	print(f"Discord bot logged in as {client.user}")
	await auto_save_loop()

@client.event
async def on_raw_message_edit(payload):
	message = payload.cached_message
	if not message:
		channel = client.get_channel(payload.channel_id)
		message = await channel.fetch_message(payload.message_id)
	
	source = f"{DISCORD_SERVER_ID}/{payload.channel_id}/{payload.message_id}"
	if message.pinned:
		if rag_model.add_document(message.content, source):
			await message.add_reaction(BOOKMARK_EMOJI)
	else:
		if rag_model.remove_document(source):
			await message.remove_reaction(BOOKMARK_EMOJI, self_member)

@tree.command(name="learn", description="我叫你記", guild=discord.Object(id=DISCORD_SERVER_ID))
@app_commands.describe(url = "訊息連結")
async def learn(interaction: discord.Interaction, url: str):
	await interaction.response.defer(ephemeral=True)
	required_ids = url.split('/')[-2:]  # [ channel_id, message_id ]
	channel = client.get_channel(int(required_ids[0]))
	message = await channel.fetch_message(int(required_ids[1]))
	
	if rag_model.add_document(message.content, f"{DISCORD_SERVER_ID}/{required_ids[0]}/{required_ids[1]}"):
		await message.add_reaction(BOOKMARK_EMOJI)
		await interaction.followup.send("已成功紀錄訊息！", ephemeral=True)
	else:
		await interaction.followup.send("紀錄訊息失敗。注意目前只能記錄文字訊息", ephemeral=True)

@tree.command(name="forget", description="這個資料我不要了", guild=discord.Object(id=DISCORD_SERVER_ID))
@app_commands.describe(url = "訊息連結")
async def forget(interaction: discord.Interaction, url: str):
	await interaction.response.defer(ephemeral=True)
	
	required_ids = url.split('/')[-2:]  # [ channel_id, message_id ]
	channel = client.get_channel(int(required_ids[0]))
	message = await channel.fetch_message(int(required_ids[1]))
	
	if rag_model.remove_document(f"{DISCORD_SERVER_ID}/{required_ids[0]}/{required_ids[1]}"):
		await message.remove_reaction(BOOKMARK_EMOJI, self_member)
		await interaction.followup.send("已成功刪除紀錄！", ephemeral=True)
	else:
		await interaction.followup.send("該訊息不存在紀錄中", ephemeral=True)

@tree.command(name="ask", description="我很好奇！", guild=discord.Object(id=DISCORD_SERVER_ID))
@app_commands.describe(question = "問題")
async def ask(interaction: discord.Interaction, question: str):
	global is_responding
	if is_responding:
		await interaction.response.send_message("有其他人正在提問，請稍後再試", ephemeral=True)
		return
		
	is_responding = True
	await interaction.response.defer(thinking=True)
	
	vars = {"msg": "", "cd": time.time()}
	async def on_token_generated(token):
		vars["msg"] += token
		if time.time() - vars["cd"] > 0.5:  # avoid frequently send http request to discord
			await interaction.edit_original_response(content=vars["msg"])
			vars["cd"] = time.time()
		
	return_message = await rag_model.generate_answer(question, on_token_generated)
	is_responding = False
	
	await interaction.edit_original_response(content=return_message)

@tree.command(name="loadallpins", description="注意！這會讀取伺服器內所有釘選訊息，可能導致 BOT 非常忙碌，請謹慎使用", guild=discord.Object(id=DISCORD_SERVER_ID))
async def load_all_pins(interaction: discord.Interaction):
	await interaction.response.defer()
	
	all_channels = interaction.guild.text_channels
	for channel in all_channels:
		print("[loadallpins] Fetching", channel.name)
		all_pins = await channel.pins()
		pin_num = len(all_pins)
		for i in range(pin_num):
			message = all_pins[i]
			source = f"{DISCORD_SERVER_ID}/{channel.id}/{message.id}"
			print("[loadallpins] Loading", source, f"({i + 1}/{pin_num})")
			
			rag_model.add_document(message.content, source)
			await message.add_reaction(BOOKMARK_EMOJI)
			
	await interaction.followup.send("所有釘選訊息載入完成！")

# start discord bot
client.run(ALL_CONFIG["discord_bot_token"])
