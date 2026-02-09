from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil, os, time
from endpoints import router
from contextlib import asynccontextmanager
from index import FaissIndex
import global_vars
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    global_vars.db_index = FaissIndex(dim=384)
    global_vars.db_index.load()
    if global_vars.db_index.index.ntotal > 0:
        print(f" 知识库加载成功！包含 {global_vars.db_index.index.ntotal} 条向量。")
    else:
        print("💡 知识库为空。")
    yield


app = FastAPI(title="YUIS", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

app.include_router(router, prefix="/api")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return FileResponse("templates/index.html")


if __name__ == "__main__":
    uvicorn.run("main:app")
