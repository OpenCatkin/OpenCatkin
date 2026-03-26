import sqlite3
import threading

class VocabDB:
    _lock = threading.Lock()

    def __init__(self, path="vocab.db"):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.c = self.conn.cursor()
        # 词表：word 唯一 → 从根源杜绝重复
        self.c.execute('''
            CREATE TABLE IF NOT EXISTS vocab (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE NOT NULL
            )
        ''')
        self.conn.commit()
        # 初始化 <unk>
        self.add_word("<unk>")

    # ==========================
    # ✅ 【核心】添加前自动校验是否存在
    # ==========================
    def add_word(self, word):
        """只有不存在时才插入，自动去重"""
        if not word or word.strip() == "":
            return False

        with self._lock:
            try:
                # 先查是否存在
                self.c.execute("SELECT id FROM vocab WHERE word = ?", (word,))
                if self.c.fetchone():
                    return False  # 已存在 → 不存

                # 不存在 → 插入
                self.c.execute("INSERT INTO vocab (word) VALUES (?)", (word,))
                self.conn.commit()
                return True
            except:
                return False

    def get_id(self, word):
        """获取ID，不存在则自动添加（带校验）"""
        self.add_word(word)
        self.c.execute("SELECT id FROM vocab WHERE word = ?", (word,))
        res = self.c.fetchone()
        return res[0] if res else 0

    def get_word(self, idx):
        """根据ID取词，安全返回"""
        try:
            self.c.execute("SELECT word FROM vocab WHERE id = ?", (idx,))
            res = self.c.fetchone()
            return res[0] if res else "<unk>"
        except:
            return "<unk>"

    def encode(self, text):
        """句子 → ID序列（自动校验+去重添加）"""
        words = text.strip().lower().split()
        return [self.get_id(w) for w in words]

    def decode(self, ids):
        """ID序列 → 句子"""
        return " ".join([self.get_word(i) for i in ids])

    def vocab_size(self):
        """词库大小"""
        self.c.execute("SELECT COUNT(*) FROM vocab")
        return self.c.fetchone()[0]