import multiprocessing as mp,os,signal,threading,time
from .database import connect,dump,load,now,one,uid,update
from .errors import ModelGuardError,require
from .services.data import get_project

_processes={};_threads={};lock=threading.Lock();_current_job=None

def recover():
    update("UPDATE jobs SET status='interrupted',stage='interrupted',error_json=?,updated_at=? WHERE status IN ('queued','running')",(dump({'code':'worker_restart','message':'Worker restarted before completion; inspect saved state before resubmitting.'}),now()))

def _execute(jid,fn,args):
    global _current_job
    _current_job=jid
    if os.name=='posix':os.setsid()
    update("UPDATE jobs SET stage='executing',progress=10,updated_at=? WHERE id=? AND status='running'",(now(),jid))
    if one("SELECT status FROM jobs WHERE id=?",(jid,))["status"]!="running":return
    try:
        result=fn(*args);update("UPDATE jobs SET status='completed',stage='completed',progress=100,result_json=?,updated_at=? WHERE id=? AND status='running'",(dump(result),now(),jid))
    except Exception as exc:
        error={'code':getattr(exc,'code','job_failed'),'message':exc.message if isinstance(exc,ModelGuardError) else 'The job failed safely; inspect input support and configured resource limits.','details':exc.details if isinstance(exc,ModelGuardError) else {'exception_type':type(exc).__name__}}
        update("UPDATE jobs SET status='failed',stage='failed',error_json=?,updated_at=? WHERE id=? AND status='running'",(dump(error),now(),jid))

def _stop(p):
    if not p.is_alive():return
    try:
        if os.name=='posix' and os.getpgid(p.pid)==p.pid:os.killpg(p.pid,signal.SIGTERM)
        else:p.terminate()
    except ProcessLookupError:pass
    p.join(1)
    if p.is_alive():p.kill();p.join()

def submit(pid,kind,fn,*args,timeout_seconds=1800):
    get_project(pid);jid=uid()
    with connect() as con:
        con.execute('BEGIN IMMEDIATE')
        if con.execute("SELECT id FROM jobs WHERE project_id=? AND status IN ('queued','running')",(pid,)).fetchone():raise ModelGuardError('A project job is already active','job_conflict',409)
        con.execute('INSERT INTO jobs(id,project_id,kind,status,created_at,updated_at,timeout_seconds) VALUES(?,?,?,?,?,?,?)',(jid,pid,kind,'queued',now(),now(),timeout_seconds))
    def work():
        p=mp.get_context('spawn').Process(target=_execute,args=(jid,fn,args),daemon=False)
        with lock:
            with connect() as con:
                con.execute('BEGIN IMMEDIATE')
                changed=con.execute("UPDATE jobs SET status='running',stage='starting',progress=1,updated_at=? WHERE id=? AND status='queued'",(now(),jid)).rowcount
            if not changed:return
            p.start();_processes[jid]=p
        start=time.monotonic()
        while p.is_alive():
            p.join(.2);record=one('SELECT status FROM jobs WHERE id=?',(jid,));status=record['status'] if record else 'cancelled'
            if status=='cancelled':_stop(p);break
            if time.monotonic()-start>timeout_seconds:
                _stop(p);update("UPDATE jobs SET status='timed_out',stage='timed_out',error_json=?,updated_at=? WHERE id=? AND status='running'",(dump({'code':'job_timeout','message':'Configured job time budget exceeded'}),now(),jid));break
        record=one('SELECT status FROM jobs WHERE id=?',(jid,))
        if record and record['status']=='running':update("UPDATE jobs SET status='failed',stage='failed',error_json=?,updated_at=? WHERE id=?",(dump({'code':'worker_exit','message':'Worker exited without a terminal result'}),now(),jid))
        with lock:_processes.pop(jid,None)
    thread=threading.Thread(target=work,daemon=True)
    with lock:_threads[jid]=thread
    thread.start();return {'job_id':jid,'status':'queued','kind':kind}

def cancel(jid):
    r=require(one('SELECT * FROM jobs WHERE id=?',(jid,)),'Job not found')
    update("UPDATE jobs SET status='cancelled',stage='cancelled',updated_at=? WHERE id=? AND status IN ('queued','running')",(now(),jid));return get(jid)
def get(jid):
    r=require(one('SELECT * FROM jobs WHERE id=?',(jid,)),'Job not found');return {'job_id':jid,'project_id':r['project_id'],'kind':r['kind'],'status':r['status'],'progress':r['progress'],'stage':r['stage'],'result':load(r['result_json']),'error':load(r['error_json']),'created_at':r['created_at'],'updated_at':r['updated_at']}


def progress(stage,percentage):
    if _current_job:update("UPDATE jobs SET stage=?,progress=?,updated_at=? WHERE id=? AND status='running'",(stage,percentage,now(),_current_job))

def shutdown():
    with lock:threads=list(_threads.items())
    for jid,thread in threads:
        thread.join(2)
        if thread.is_alive():
            with lock:process=_processes.get(jid)
            if process:
                update("UPDATE jobs SET status='interrupted',stage='interrupted',error_json=?,updated_at=? WHERE id=? AND status='running'",(dump({'code':'worker_shutdown','message':'Worker stopped before completion'}),now(),jid))
                _stop(process)
            thread.join(2)
        with lock:_threads.pop(jid,None)
