import json
import string
import pickle
import boto3
from kafka import KafkaConsumer, KafkaProducer
from pythainlp import word_tokenize, subword_tokenize
from scipy.sparse import hstack
from itertools import chain
import os
from dotenv import load_dotenv
load_dotenv()

CONSUMER_TOPIC_NAME = "QUERY"
PRODUCER_TOPIC_NAME = "QUERY-RESPONSE"
KAFKA_SERVER = "localhost:9092"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    api_version=(0, 11, 15)
)

consumer = KafkaConsumer(
    CONSUMER_TOPIC_NAME,
    bootstrap_servers=KAFKA_SERVER,
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
)

s3 = boto3.resource(
    service_name='s3',
    region_name=os.getenv('AWS_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)
class Diagnose:
    @staticmethod
    def download_s3_folder(bucket_name, s3_folder, local_dir=None):
        bucket = s3.Bucket(bucket_name)
        try:
            for obj in bucket.objects.filter(Prefix=s3_folder):
                target = obj.key if local_dir is None \
                    else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
                if not os.path.exists(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target))
                if obj.key[-1] == '/':
                    continue
                bucket.download_file(obj.key, target)
            return "download completed"
        except:
            return "download failed"


def text_processor(text, whitespace=True):
    text = [w.lower() for w in word_tokenizer(text, whitespace)]
    text = [word.translate(str.maketrans('', '', string.punctuation + u'\xa0')) for word in text]
    # NOTE: Remove number from text ***may be used
    # text = [word for word in text if not word.isnumeric()]
    text = ''.join(text)
    return text

def word_tokenizer(text , whitespace=False):
    token_word = word_tokenize(text=text , keep_whitespace=whitespace, custom_dict=trie)
    return token_word

def syllable_tokenizer(text , whitespace=False):
    syllable_word = subword_tokenize(text, engine='ssg', keep_whitespace=whitespace)
    syllable_word = [word_tokenizer(w, whitespace) for w in syllable_word]
    syllable_word = list(chain.from_iterable(syllable_word))
    return syllable_word

if __name__ == '__main__':
    Diagnose.download_s3_folder('cds-bucket', 'pickles')
    # NOTE: Download pickle file from s3
    trie = pickle.load(open('pickles/trie.pkl', 'rb'))
    models = pickle.load(open('pickles/models.pkl', 'rb'))
    word_vec = pickle.load(open('pickles/word_vec.pkl', 'rb'))
    syllable_vec = pickle.load(open('pickles/syllable_vec.pkl', 'rb'))
    topic_vec = pickle.load(open('pickles/topic_vec.pkl', 'rb'))
    print('Load file successfully')

    for msg in consumer:
        print('Diagnose process is started')
        tv = word_vec.transform([msg.value['symptom']])
        sv = syllable_vec.transform([msg.value['symptom']])
        tpv = topic_vec.transform([msg.value['symptom']])
        query_vec = hstack([tv, sv, tpv])

        results = {}
        for m in models:
            results[m['part']] = m['model'].predict_proba(query_vec)[0][1]
        result = dict(sorted(results.items(), key=lambda x: x[1], reverse=True)[:5])
        # Kafka produce
        json_payload = json.dumps(result)
        json_payload = str.encode(json_payload)

        producer.send(PRODUCER_TOPIC_NAME, json_payload)
        producer.flush()