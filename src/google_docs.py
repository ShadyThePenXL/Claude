from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/documents"]


class GoogleDocsClient:
    def __init__(self, credentials_file: str, document_id: str):
        self.document_id = document_id
        self._service = None
        self._enabled = False

        if not document_id or not credentials_file:
            return

        creds_path = Path(credentials_file)
        if not creds_path.exists():
            return

        try:
            creds = service_account.Credentials.from_service_account_file(
                str(creds_path), scopes=SCOPES
            )
            self._service = build("docs", "v1", credentials=creds)
            self._enabled = True
        except Exception:
            pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    def read_doc(self) -> str:
        if not self._enabled:
            return ""

        doc = self._service.documents().get(documentId=self.document_id).execute()
        content = doc.get("body", {}).get("content", [])
        text_parts = []
        for element in content:
            if "paragraph" in element:
                for run in element["paragraph"].get("elements", []):
                    text_run = run.get("textRun")
                    if text_run:
                        text_parts.append(text_run["content"])
        return "".join(text_parts)

    def append_to_doc(self, text: str) -> None:
        if not self._enabled:
            return

        doc = self._service.documents().get(documentId=self.document_id).execute()
        content = doc.get("body", {}).get("content", [])
        end_index = content[-1]["endIndex"] if content else 1

        requests = [
            {
                "insertText": {
                    "location": {"index": end_index - 1},
                    "text": text + "\n\n",
                }
            }
        ]
        self._service.documents().batchUpdate(
            documentId=self.document_id, body={"requests": requests}
        ).execute()
