from __future__ import annotations

from urllib.parse import quote

from ..utils.comfy_paths import get_output_directory
from ..utils.path_utils import resolve_download_path

_ROUTES_REGISTERED = False


def register_download_routes() -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return

    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return

    routes = PromptServer.instance.routes

    @routes.get("/arata-joycaption-shots/download")
    async def download_description_file(request):
        relative_path = str(request.query.get("path", "")).strip()
        if not relative_path:
            return web.Response(status=400, text="Missing required 'path' query parameter.")

        try:
            output_root = get_output_directory()
            file_path = resolve_download_path(relative_path, output_root)
        except ValueError as exc:
            return web.Response(status=400, text=str(exc))

        if not file_path.is_file():
            return web.Response(status=404, text=f"File not found: {relative_path}")

        response = web.FileResponse(path=file_path)
        quoted_name = quote(file_path.name)
        content_type = (
            "application/json; charset=utf-8"
            if file_path.suffix.lower() == ".json"
            else "text/plain; charset=utf-8"
        )
        response.headers["Content-Type"] = content_type
        response.headers["Content-Disposition"] = (
            f"attachment; filename=\"{file_path.name}\"; filename*=UTF-8''{quoted_name}"
        )
        return response

    _ROUTES_REGISTERED = True
