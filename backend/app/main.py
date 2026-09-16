from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from .config import settings
from .database import init_db, load, one
from .errors import ModelGuardError
from .jobs import submit
from .schemas import ConfigureRequest, DataTwinRequest, DriftRequest, ExperimentRequest, ProjectCreate, ShieldRequest
from .services import audit as audit_service
from .services import drift, experiments, modeling, reliability, shield, twin
from .services.data import configure, create_project, summarize, upload_dataset


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app=FastAPI(title="ModelGuard API",version="0.1.0",description="Evidence-first ML reliability, failure analysis, scenario testing, and batch drift monitoring.",lifespan=lifespan)


@app.exception_handler(ModelGuardError)
async def modelguard_error(_:Request,exc:ModelGuardError):
    return JSONResponse(status_code=exc.status_code,content={"error":{"code":exc.code,"message":exc.message,"details":exc.details}})


@app.exception_handler(Exception)
async def unexpected(_:Request,exc:Exception):
    return JSONResponse(status_code=500,content={"error":{"code":"internal_error","message":"An internal error occurred."}})


@app.get("/health")
def health(): return {"status":"ok"}


@app.post("/projects",status_code=201)
def projects_create(body:ProjectCreate): return create_project(body.name,body.description)


@app.post("/projects/{project_id}/datasets",status_code=201)
async def datasets_upload(project_id:str,file:UploadFile=File(...)):
    content=await file.read(settings.max_upload_bytes+1)
    return upload_dataset(project_id,file.filename or "",content)


@app.get("/projects/{project_id}/datasets/{dataset_id}/summary")
def datasets_summary(project_id:str,dataset_id:str): return summarize(project_id,dataset_id)


@app.post("/projects/{project_id}/configure")
def projects_configure(project_id:str,body:ConfigureRequest): return configure(project_id,body.model_dump())


@app.post("/projects/{project_id}/audit",status_code=202)
def projects_audit(project_id:str): return submit(project_id,"audit",audit_service.audit,project_id)


@app.post("/projects/{project_id}/models/train",status_code=202)
def models_train(project_id:str): return submit(project_id,"model_training",modeling.train,project_id)


@app.get("/projects/{project_id}/models/results")
def models_results(project_id:str): return modeling.model_results(project_id)


@app.post("/projects/{project_id}/modelshield/analyze",status_code=202)
def modelshield_analyze(project_id:str,body:ShieldRequest): return submit(project_id,"modelshield",shield.analyze,project_id,body.model_dump())


@app.post("/projects/{project_id}/datatwin/simulate",status_code=202)
def datatwin_simulate(project_id:str,body:DataTwinRequest): return submit(project_id,"datatwin",twin.simulate,project_id,body.model_dump())


@app.post("/projects/{project_id}/experiments",status_code=202)
def experiments_run(project_id:str,body:ExperimentRequest): return submit(project_id,"experiment",experiments.run,project_id,body.model_dump())


@app.post("/projects/{project_id}/drift/analyze",status_code=202)
def drift_analyze(project_id:str,body:DriftRequest): return submit(project_id,"drift",drift.analyze,project_id,body.model_dump())


@app.get("/projects/{project_id}/reliability")
def reliability_get(project_id:str,include_indicator:bool=Query(False)): return reliability.assessment(project_id,include_indicator)


@app.get("/projects/{project_id}/report")
def report_get(project_id:str,include_indicator:bool=Query(False)): return reliability.report(project_id,include_indicator)


@app.get("/jobs/{job_id}")
def jobs_get(job_id:str):
    row=one("SELECT * FROM jobs WHERE id=?",(job_id,))
    if not row: raise ModelGuardError("Job not found","not_found",404)
    return {"job_id":row["id"],"project_id":row["project_id"],"kind":row["kind"],"status":row["status"],"result":load(row["result_json"]),"error":load(row["error_json"]),"created_at":row["created_at"],"updated_at":row["updated_at"]}
