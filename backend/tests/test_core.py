import io
import time

import numpy as np
import pandas as pd


def wait(client,jid):
    for _ in range(2400):
        result=client.get(f"/jobs/{jid}").json()
        if result["status"] in {"completed","failed"}: return result
        time.sleep(.05)
    raise AssertionError("job did not finish")


def setup_project(client,n=240):
    rng=np.random.default_rng(7); age=rng.normal(40,12,n); region=rng.choice(["north","south","west"],n); y=((age>42)&(region!="west")).astype(int); y[rng.random(n)<.12]^=1
    df=pd.DataFrame({"age":age,"income":age*1000+rng.normal(0,5000,n),"region":region,"target":y})
    pid=client.post("/projects",json={"name":"test"}).json()["id"]
    up=client.post(f"/projects/{pid}/datasets",files={"file":("sample.csv",df.to_csv(index=False),"text/csv")})
    assert up.status_code==201; did=up.json()["id"]
    cfg=client.post(f"/projects/{pid}/configure",json={"dataset_id":did,"target_column":"target","task":"auto","primary_metric":"macro_f1","max_trials":2,"weight_trials":4,"fit_timeout_seconds":30})
    assert cfg.status_code==200
    return pid,did


def test_upload_rejects_non_csv_and_path_traversal(client):
    pid=client.post("/projects",json={"name":"x"}).json()["id"]
    assert client.post(f"/projects/{pid}/datasets",files={"file":("../evil.txt","a,b\n1,2","text/plain")}).status_code==400


def test_end_to_end_training_and_reliability(client):
    pid,did=setup_project(client)
    audit=wait(client,client.post(f"/projects/{pid}/audit").json()["job_id"]); assert audit["status"]=="completed"
    trained=wait(client,client.post(f"/projects/{pid}/models/train").json()["job_id"]); assert trained["status"]=="completed",trained
    result=trained["result"]; assert result["description"].startswith("The best-performing baseline")
    assert "macro_f1" in result["metrics"]
    shield=wait(client,client.post(f"/projects/{pid}/modelshield/analyze",json={"min_samples":10,"min_support":.05,"bootstrap_iterations":20}).json()["job_id"]); assert shield["status"]=="completed",shield
    rel=client.get(f"/projects/{pid}/reliability").json(); assert next(d for d in rel["dimensions"] if d["dimension"]=="Evaluation Integrity")["status"]=="Pass"; assert "provisional_indicator" not in rel


def test_unlabelled_scenario_omits_performance(client):
    pid,_=setup_project(client); trained=wait(client,client.post(f"/projects/{pid}/models/train").json()["job_id"]); assert trained["status"]=="completed"
    body={"include_labels":False,"scenarios":[{"type":"missingness","feature":"age","magnitude":.2}]}
    sim=wait(client,client.post(f"/projects/{pid}/datatwin/simulate",json=body).json()["job_id"]); assert sim["status"]=="completed",sim
    item=sim["result"]["results"][0]; assert item["performance_metrics"] is None; assert "Controlled scenario" in item["label"]
