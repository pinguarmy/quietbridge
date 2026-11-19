"""Raw camera-session chunks; computation windows are not sentence boundaries."""


class SessionTranscript:
    def __init__(self):
        self.chunks = []

    @property
    def text(self):
        return " ".join(self.chunks)

    def append(self, text):
        text = text.strip()
        if text:
            self.chunks.append(text)
        return self.text

    def reset(self):
        self.chunks.clear()
