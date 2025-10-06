import re
from typing import Optional, Union, Literal, Any
from langchain_text_splitters.base import TextSplitter


def _split_text_with_regex(
    text: str, separator: str, *, keep_separator: Union[bool, Literal["start", "end"]]
) -> list[str]:
    """Split text by regex, optionally keeping the separator."""
    if separator:
        if keep_separator:
            # Keep separator in result
            _splits = re.split(f"({separator})", text)
            if keep_separator == "end":
                splits = [_splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)]
                if len(_splits) % 2 == 0:
                    splits += _splits[-1:]
            else:  # "start"
                splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
                if _splits[0]:
                    splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class RecursiveCharacterTextSplitter(TextSplitter):
    """Splitting text by recursively looking at characters."""

    def __init__(
        self,
        separators: Optional[list[str]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = True,
        is_separator_regex: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        # initialize parent
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex

        # allow overrides
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function or self.length_function

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, keep_separator=self._keep_separator)

        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self.merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    final_chunks.extend(self._split_text(s, new_separators))

        if _good_splits:
            merged_text = self.merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)

        return final_chunks

    def split_text(self, text: str) -> list[str]:
        return self._split_text(text, self._separators)

    # ---------------- OVERRIDES ----------------
    def merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Greedily merge small splits into chunks not exceeding chunk_size."""
        if not splits:
            return []

        chunks: list[str] = []
        current = splits[0]
        max_size = int(self._chunk_size)
        sep = separator or ""

        for part in splits[1:]:
            candidate = current + (sep + part if current and sep else part)
            if self._length_function(candidate) <= max_size:
                current = candidate
            else:
                chunks.append(current)
                current = part
        chunks.append(current)
        return chunks

    @staticmethod
    def length_function(text: str) -> int:
        """Default length function (character count)."""
        return len(text)


# ---------------- TEST EXAMPLE ----------------
if __name__ == "__main__":
    sample_text = """What I Worked On

    February 2021

    Before college the two main things I worked on, outside of school,
    were writing and programming. I didn't write essays. I wrote what
    beginning writers were supposed to write then, and probably still are:
    short stories. My stories were awful.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=10,
        keep_separator=False,
    )

    chunks = splitter.split_text(sample_text)

    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"[{i}] len={len(chunk)} -> {repr(chunk)}")
