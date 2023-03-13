import json
from locust import HttpUser, task, between
import pandas as pd

test_data = pd.read_excel("../../place_tm_20230216_053926.xlsx")


class PerformanceTests(HttpUser):
    # wait_time = between(1, 3)

    @task(1)
    def testPredict(self):
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        text = {"text": test_data.sample(1)['dst_content'].values[0]}
        data = json.dumps(text)
        self.client.post("/api/langid", data=data, headers=headers)

    @task(1)
    def testPredict_batch(self):
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        text = {"text": test_data.sample(32)['dst_content'].values.tolist()}
        data = json.dumps(text)
        self.client.post("/api/langid", data=data, headers=headers)