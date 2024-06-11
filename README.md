# Discord AI 助理機器人
採用 [RAG](https://blog.infuseai.io/rag-retrieval-augmented-generation-introduction-a5854cb6393e) 架構的語言模型進行資料檢索與回答問題
能紀錄 Discord 內特定訊息，並在使用者提出問題時參考相關資料進行回答，同時附上參考訊息連結
目前沒有紀錄歷史訊息的功能，若要讓 AI 參考過去的對話內容，需自行將歷史訊息中的必要資訊包含在提出的問題中

## 安裝方式
1. 將 `config.template` 更名為 `config` 並設定相關參數
2. 將 `prompt.template` 更名為 `prompt`
    - 這是實際向 AI 提問的內容格式
    - 你可以自行修改內容，但必須包含 {context} 與 {question} 兩項參數
    - context 會被替換為與使用者提問相關的參考訊息
    - question 會被替換為使用者的問題
3. 安裝 3.10+ 版本的 [python](https://www.python.org/downloads/)
4. 開啟 cmd 輸入以下指令安裝相關套件
```
# pip install langchain
# pip install langchain_community
# pip install -U sentence-transformers
# pip install chardet
# pip install chromadb
# pip install discord.py
```
5. 安裝 [Ollama](https://www.ollama.com/)
6. 開啟 cmd 輸入 `ollama pull <model_name>`，其中 model name 需與 config 中的 `ollama_model` 參數相同
7. 雙擊 `server.py` 或在此路徑下開啟 cmd 視窗並輸入 `python server.py` 執行
8. 在 Discord 伺服器中給予機器人對應的身分組，使其有權限讀取必要頻道的訊息

## 功能說明
- 執行上述 `server.py` 並看到 `Discord bot logged in as ...` 訊息即為啟動成功。此時在邀請了機器人的 Discord 伺服器，應該可以看到機器人顯示狀態為 `線上`
- 預設狀態下有訊息被 `釘選`/`取消釘選` 的話，機器人就會將該訊息 `紀錄到資料庫`/`從資料庫刪除`
    - 當釘選訊息有變動時，會同步更新資料庫
    - 可透過 config 關閉此功能

當使用者提問時，會將資料庫中相關的資料交給語言模型參考後進行回答
當機器人紀錄訊息時，會對訊息加上 📑 的反應，並在刪除紀錄時移除反應

### 指令列表
- `/ask <question>`：向 AI 提出問題
    - 目前為避免機器忙碌，有限制一次只能回答一個人的問題，在 AI 回答過程中提出問題會被阻擋
- `/learn <url>`：強制將指定訊息記錄到資料庫中
    - 傳入的參數為訊息連結，對訊息點擊右鍵    -> 複製訊息連結 即可取得
    - 若已紀錄的訊息內容有變動，會同步更新資料庫
    - 透過該指令紀錄的訊息，即使被取消釘選也不會從資料庫中刪除
- `/forget <url>`：強制將指定訊息從資料庫中刪除
    - 傳入的參數為訊息連結，對訊息點擊右鍵    -> 複製訊息連結 即可取得
- `/forgetchannel`：將輸入指令時所在頻道中，所有被記錄的訊息從資料庫中移除
    - 通常於無用頻道刪除前使用
    - 也可於 `loadallpins` 後將不重要頻道的訊息移除
- `/loadallpins`：掃描所有伺服器內的頻道，將釘選訊息全部記錄到資料庫中
    - 該指令基本只需要在初次加入伺服器時使用，後續都是根據釘選事件或指令執行相關資料庫操作
    - 該指令會花費大量時間掃描訊息，請特別注意使用