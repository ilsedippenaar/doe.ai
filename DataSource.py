from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from shutil import copyfileobj, move, rmtree
from tempfile import SpooledTemporaryFile
from re import sub

_dataDir = Path("data")
if not _dataDir.exists():
    _dataDir.mkdir()


class _AbstractDataSource(ABC):
    @property
    @abstractmethod
    def _url(self):
        return ""

    @property
    @abstractmethod
    def _localPath(self):
        return ""

    @abstractmethod
    def getData(self):
        pass


class _AliceText(_AbstractDataSource):
    @property
    def _localPath(self):
        return _dataDir / "alice_text"

    @property
    def _url(self):
        return "https://ia801405.us.archive.org/18/items/alicesadventures19033gut/19033.txt"

    def getData(self):
        if not self._localPath.exists():
            with urlopen(self._url) as response:
                data = response.read().decode("ascii")
            data = data.split('\r\n')[76:-371]  # this is just hardcoded to remove extraneous lines from the data
            data = [line for line in data if line != "" and "*" not in line and "[Illustration]" not in line]
            data = sub(r"_([\w\']+)?_", r"\1", " ".join(data))  # eliminates any underscores around a word
            # TODO: make all data lower case and eliminate punctuation for easier word labelling
            with open(self._localPath, 'w') as f:
                f.write(data)

        with open(self._localPath, 'r') as f:
            data = f.readline().split()
        return data


class _CornellMovieCorpus(_AbstractDataSource):
    @property
    def _localPath(self):
        return _dataDir / "cornell_movie_corpus"

    @property
    def _url(self):
        return "http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip"

    def getData(self):
        if not self._localPath.exists():
            rootZipDir = "cornell movie-dialogs corpus"
            with urlopen(self._url) as response, SpooledTemporaryFile() as tmp:
                copyfileobj(response, tmp)
                with ZipFile(tmp) as zipTmp:
                    infoList = zipTmp.infolist()
                    for info in infoList:
                        pathFile = Path(info.filename)
                        if pathFile.parts[0] == rootZipDir and pathFile.stem != ".DS_Store":
                            zipTmp.extract(info, self._localPath)
            for p in self._localPath.joinpath(rootZipDir).iterdir():
                move(str(p), str(self._localPath))
            rmtree(str(self._localPath / rootZipDir))

        return ""


class DataSource(Enum):
    ALICE_TEXT = _AliceText()
    CORNELL_MOVIE_CORPUS = _CornellMovieCorpus()
    CORNELL_MOVIE_QUOTES_CORPUS = "https://www.cs.cornell.edu/~cristian/memorability_files/cornell_movie_quotes_corpus.zip"
