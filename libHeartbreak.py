
# from transformers import MT5ForConditionalGeneration, T5Tokenizer
from pythainlp.summarize import summarize
from pythainlp.summarize import extract_keywords
import re
from transformers import pipeline
import torch
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pythainlp.tokenize import sent_tokenize
import hdbscan
from collections import Counter
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

MIN_LENGTH = 100


def clean(text: str) -> str:
    if not text:
        return ""
    
    # แปลง newline/tab เป็น space
    text = text.replace("\n", " ").replace("\t", " ")
    
    # ลบสัญลักษณ์พิเศษ (ยกเว้นตัวอักษรไทย/อังกฤษ/ตัวเลข)
    text = re.sub(r"[^ก-๙a-zA-Z0-9\s]", "", text)
    
    # ลบ space เกิน
    text = re.sub(r"\s+", " ", text)
    
    # Trim ขอบข้อความ
    text = text.strip()
    
    return text


def summarize_conditional(text: str) -> str:
    cleaned_text = clean(text)  # เรียก clean ก่อน
    MIN_LENGTH = 100
    if len(cleaned_text) < MIN_LENGTH:
        return cleaned_text
    summary_result = summarize(cleaned_text, engine="mt5-cpe-kmutt-thai-sentence-sum")
    if isinstance(summary_result, list):
        summary_result = " ".join(summary_result)

    return summary_result



# 1. Sentence embedding model
embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 2. Zero-shot model (ช้า แต่แม่น)
classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli",
    device=0  # ถ้ามี GPU จะเร็วขึ้นมาก
)

# cache สำหรับประโยคซ้ำ
reason_cache = {}

reason_labels = [
    "ปัญหาการสื่อสาร",
    "ความต้องการไม่ตรงกัน",
    "ความไว้วางใจและความหึงหวง",
    "การนอกใจหรือมือที่สาม",
    "ปัญหาทางเพศ",
    "ความคาดหวังไม่ตรงกัน",
    "ปัญหาทางอารมณ์หรือความเครียด",
    "ความสัมพันธ์จืดจางหรือเบื่อหน่าย",
    "ปัญหาทางการเงิน",
    "ค่านิยมและเป้าหมายชีวิตไม่ตรงกัน",
    "ครอบครัวหรือคนรอบข้างแทรกแซง",
    "การจัดการความขัดแย้งไม่ดี",
    "เวลาที่มีให้กันไม่พอ",
    "ไลฟ์สไตล์ไม่ตรงกัน",
    "ภาระงานหรือการเรียนกระทบความสัมพันธ์",
    "ข้อจำกัดทางระยะทาง (LDR / อยู่ไกลกัน)",
    "บุคลิกเข้ากันไม่ได้",
    "ความไม่สม่ำเสมอในการแสดงความรัก",
    "ความคาดหวังเรื่องอนาคต (แต่งงาน / มีลูก)",
    "ความไม่มั่นคงทางความรู้สึกส่วนตัว (self-esteem)",
    "ปัญหาการควบคุมหรือความเป็นเจ้าของมากเกินไป",
    "พฤติกรรมเสพติด (เกม แอลกอฮอล์ การพนัน โทรศัพท์)",
    "ปัญหาการจัดการเวลาและสมดุลชีวิต",
    "ความต่างด้านศาสนา วัฒนธรรม หรือพื้นฐานครอบครัว",
]

def detect_reason_batch(sentences):
    """ทำนายเหตุผลแบบ batch เพื่อให้เร็วขึ้นมาก"""
    uncached = [s for s in sentences if s not in reason_cache]
    
    if uncached:
        results = classifier(uncached, reason_labels)
        for s, r in zip(uncached, results):
            reason_cache[s] = r["labels"][0]

    return [reason_cache[s] for s in sentences]


def cluster_reasons(text: str):

    print("\n🔎 ตัดประโยค...")
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return {"global_reason": detect_reason_batch([text])[0], "sentences": sentences}

    print("📌 จำนวนประโยค:", len(sentences))

    print("\n🔧 กำลังสร้าง embedding ...")
    embeddings = embedder.encode(sentences, show_progress_bar=True)

    print("\n🔍 กำลังทำ HDBSCAN clustering ...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)

    # fallback ถ้า HDBSCAN ไม่เจอ cluster
    if len(set(labels)) <= 1:
        print("\n⚠️ HDBSCAN ไม่เจอ cluster → ใช้ Agglomerative แทน")
        # from sklearn.cluster import AgglomerativeClustering
        # n_clusters = min(3, len(sentences))
        # clusterer2 = AgglomerativeClustering(n_clusters=n_clusters)
        # labels = clusterer2.fit_predict(embeddings)

    print("\n📊 ทำนายเหตุผลของแต่ละ cluster ...")

    result = {}

    for label in set(labels):
        cluster_sents = [sentences[i] for i in range(len(sentences)) if labels[i] == label]
        
        reasons = detect_reason_batch(cluster_sents)
        main_reason = Counter(reasons).most_common(1)[0][0]

        result[main_reason] = cluster_sents

    return result



emotion_labels = [
    "ตลก/ประชด",
    "เศร้า/เสียใจ",
    "โกรธ/ระบาย",
    "รุนแรง/อันตราย",
    "สับสน",
    "คิดถึง",
    "ปลง/ยอมรับ"
]

# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
label_embs = embedder.encode(emotion_labels)

# ---------- FUNCTION ----------
def predict_emotion(text: str):

    if not isinstance(text, str) or text.strip() == "":
        return None, None

    text_emb = embedder.encode(text)
    sims = cosine_similarity([text_emb], label_embs)[0]
    best_idx = np.argmax(sims)

    return emotion_labels[best_idx], float(sims[best_idx])