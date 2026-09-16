from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .database import dump, insert, now, one, uid, update

executor=ThreadPoolExecutor(max_workers=2,thread_name_prefix="modelguard")


def submit(project_id,kind,fn,*args):
    jid=uid(); insert("jobs",{"id":jid,"project_id":project_id,"kind":kind,"status":"queued","result_json":None,"error_json":None,"created_at":now(),"updated_at":now()})
    def work():
        update("UPDATE jobs SET status=?,updated_at=? WHERE id=?",("running",now(),jid))
        try:
            result=fn(*args); update("UPDATE jobs SET status=?,result_json=?,updated_at=? WHERE id=?",("completed",dump(result),now(),jid))
        except Exception as exc:
            error={"code":getattr(exc,"code","job_failed"),"message":getattr(exc,"message",str(exc))}; update("UPDATE jobs SET status=?,error_json=?,updated_at=? WHERE id=?",("failed",dump(error),now(),jid))
    executor.submit(work); return {"job_id":jid,"status":"queued","kind":kind}
