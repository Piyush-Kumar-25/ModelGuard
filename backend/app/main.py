from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Query, Request, UploadFile, Form
from fastapi.responses import JSONResponse, Response

from .config import settings
from .database import init_db, load, one
from .errors import ModelGuardError
from .jobs import submit, recover, shutdown
from .schemas import ConfigureRequest, DataTwinRequest, DriftRequest, ExperimentRequest, ProjectCreate, ShieldRequest
from .services import audit as audit_service
from .services import drift, experiments, modeling, reliability, shield, twin
from .services.data import configure, create_project, summarize, upload_dataset


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    recover()
    yield
    shutdown()


app=FastAPI(title="ModelGuard API",version="1.0.0",description="Evidence-first ML reliability, failure analysis, scenario testing, and batch drift monitoring.",lifespan=lifespan)


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
    from .jobs import get
    return get(job_id)

@app.post("/jobs/{job_id}/cancel")
def jobs_cancel(job_id:str):
    from .jobs import cancel
    return cancel(job_id)

from .schemas import MappingRequest, PredictRequest, LifecycleRequest
from .services import exporting
from .services.data import suggestions
from fastapi.exceptions import RequestValidationError
import pandas as pd

@app.exception_handler(RequestValidationError)
async def invalid_request(_:Request,exc:RequestValidationError):
    return JSONResponse(status_code=422,content={"error":{"code":"validation_error","message":"Invalid request","details":[{"location":list(e['loc']),"message":e['msg']} for e in exc.errors()]}})

@app.get("/projects/{project_id}/datasets/{dataset_id}/target-suggestions")
def targets_suggest(project_id:str,dataset_id:str):return suggestions(project_id,dataset_id)

@app.post("/projects/{project_id}/monitoring/batches",status_code=201)
async def monitoring_upload(project_id:str,file:UploadFile=File(...),model_id:str|None=Form(None),batch_timestamp:str|None=Form(None),record_id_column:str|None=Form(None)):
    return drift.upload_batch(project_id,file.filename or '',await file.read(settings.max_upload_bytes+1),model_id,batch_timestamp,record_id_column)

@app.get("/projects/{project_id}/monitoring/batches")
def monitoring_batches(project_id:str):return drift.list_batches(project_id)

@app.post("/projects/{project_id}/monitoring/batches/{batch_id}/mapping")
def monitoring_mapping(project_id:str,batch_id:str,body:MappingRequest):return drift.confirm_mapping(project_id,batch_id,body.model_dump())

@app.post("/projects/{project_id}/monitoring/batches/{batch_id}/analyze",status_code=202)
def monitoring_analyze(project_id:str,batch_id:str):return submit(project_id,'drift',drift.analyze,project_id,{'dataset_id':batch_id})

@app.post("/projects/{project_id}/monitoring/batches/{batch_id}/labels",status_code=201)
async def monitoring_labels(project_id:str,batch_id:str,file:UploadFile=File(...),target_column:str|None=Form(None)):
    return drift.attach_labels(project_id,batch_id,file.filename or '',await file.read(settings.max_upload_bytes+1),target_column)

@app.get("/projects/{project_id}/monitoring/history")
def monitoring_history(project_id:str,model_id:str|None=None):return drift.history(project_id,model_id)

@app.post("/projects/{project_id}/models/{model_id}/lifecycle")
def model_lifecycle(project_id:str,model_id:str,body:LifecycleRequest):return experiments.lifecycle(project_id,model_id,body.action)

@app.get("/projects/{project_id}/models/{model_id}/card")
def model_card(project_id:str,model_id:str,format:str=Query('json',pattern='^(json|markdown)$')):
    data=exporting.card(project_id,model_id)
    if format=='json':return JSONResponse(data,headers={'Content-Disposition':'attachment; filename="model_card.json"'})
    return Response(exporting.markdown('ModelGuard Model Card',data),media_type='text/markdown',headers={'Content-Disposition':'attachment; filename="model_card.md"'})

@app.get("/projects/{project_id}/models/{model_id}/report")
def model_report(project_id:str,model_id:str,format:str=Query('json',pattern='^(json|markdown)$')):
    data=reliability.report(project_id,model_id=model_id)
    if format=='json':return JSONResponse(data,headers={'Content-Disposition':'attachment; filename="reliability_report.json"'})
    return Response(exporting.markdown('ModelGuard Reliability Report',data),media_type='text/markdown',headers={'Content-Disposition':'attachment; filename="reliability_report.md"'})

@app.get("/projects/{project_id}/models/{model_id}/export")
def model_export(project_id:str,model_id:str):
    data,name=exporting.bundle(project_id,model_id)
    return Response(data,media_type='application/zip',headers={'Content-Disposition':f'attachment; filename="{name}"'})

@app.post("/projects/{project_id}/models/{model_id}/predict")
def model_predict(project_id:str,model_id:str,body:PredictRequest):
    model,art=modeling.load_model(project_id,model_id);frame=pd.DataFrame(body.records);check=drift.schema_check(frame,art['config']);x=drift.model_inputs(frame,art['config'],check);pipe=art['pipeline'];result={'model_id':model_id,'version':model['version'],'schema_compatibility':check,'predictions':pipe.predict(x).tolist()}
    if model['task']=='classification':result.update(probabilities=pipe.predict_proba(x).tolist(),class_order=pipe.classes_.tolist())
    return result
