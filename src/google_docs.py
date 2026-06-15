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

    def _extract_text(self, content: list) -> str:
        text_parts = []
        for element in content:
            if "paragraph" in element:
                for run in element["paragraph"].get("elements", []):
                    text_run = run.get("textRun")
                    if text_run:
                        text_parts.append(text_run["content"])
        return "".join(text_parts)

    def _get_doc(self) -> dict:
        return (
            self._service.documents()
            .get(documentId=self.document_id, includeTabsContent=True)
            .execute()
        )

    def get_tabs(self) -> dict[str, str]:
        if not self._enabled:
            return {}
        doc = self._get_doc()
        result = {}
        for tab in doc.get("tabs", []):
            props = tab.get("tabProperties", {})
            title = props.get("title", "")
            tab_id = props.get("tabId", "")
            if title:
                result[title] = tab_id
        return result

    def read_doc(self) -> str:
        if not self._enabled:
            return ""
        doc = self._get_doc()
        tabs = doc.get("tabs", [])
        if tabs:
            all_text = []
            for tab in tabs:
                title = tab.get("tabProperties", {}).get("title", "")
                content = (
                    tab.get("documentTab", {}).get("body", {}).get("content", [])
                )
                text = self._extract_text(content)
                if text.strip():
                    all_text.append(f"[{title}]\n{text}")
            return "\n\n".join(all_text)
        content = doc.get("body", {}).get("content", [])
        return self._extract_text(content)

    def read_tab_by_name(self, name: str) -> str:
        if not self._enabled:
            return ""
        doc = self._get_doc()
        name_lower = name.lower()
        for tab in doc.get("tabs", []):
            title = tab.get("tabProperties", {}).get("title", "")
            if name_lower in title.lower() or title.lower() in name_lower:
                content = (
                    tab.get("documentTab", {}).get("body", {}).get("content", [])
                )
                return self._extract_text(content)
        return ""

    def append_to_doc(self, text: str) -> None:
        if not self._enabled:
            return
        doc = self._get_doc()
        tabs = doc.get("tabs", [])
        if tabs:
            tab = tabs[0]
            tab_id = tab.get("tabProperties", {}).get("tabId", "")
            content = (
                tab.get("documentTab", {}).get("body", {}).get("content", [])
            )
            end_index = content[-1]["endIndex"] if content else 1
            requests = [
                {
                    "insertText": {
                        "location": {"index": end_index - 1, "tabId": tab_id},
                        "text": text + "\n\n",
                    }
                }
            ]
        else:
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

    def append_to_tab(self, tab_id: str, text: str) -> None:
        if not self._enabled:
            return
        doc = self._get_doc()
        for tab in doc.get("tabs", []):
            if tab.get("tabProperties", {}).get("tabId") == tab_id:
                content = (
                    tab.get("documentTab", {}).get("body", {}).get("content", [])
                )
                end_index = content[-1]["endIndex"] if content else 1
                requests = [
                    {
                        "insertText": {
                            "location": {
                                "index": end_index - 1,
                                "tabId": tab_id,
                            },
                            "text": text + "\n\n",
                        }
                    }
                ]
                self._service.documents().batchUpdate(
                    documentId=self.document_id, body={"requests": requests}
                ).execute()
                return
