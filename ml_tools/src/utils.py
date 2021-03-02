import json
import datetime
from pathlib import Path
from typing import Dict, Any


class GenericObjJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError as ex:
            return str(obj)

def create_run_id() -> str:
    """Creates a run ID based on the month, day & time"""
    id = datetime.datetime.now().strftime("%m%d_%H%M")
    return id

def write_to_json(data: Dict, ffp: Path):
    """Writes the data to the specified file path"""
    if ffp.is_dir():
        raise FileExistsError(f"File {ffp} already exists, failed to save the config!")

    with open(ffp, "w") as f:
        json.dump(data, f, cls=GenericObjJSONEncoder, indent=4)

def load_json(ffp: Path):
    """Loads a data from a json file"""
    with open(ffp, "r") as f:
        return json.load(f)

