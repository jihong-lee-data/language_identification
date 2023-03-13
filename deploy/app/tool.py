import json
import os
import re
from pathlib import Path


def load_json(path):
    """
    Load a JSON file from a given path.

    Args:
    - path: string - The path of the JSON file to load

    Returns:
    - dict - The dictionary of the loaded JSON file
    """
    with open(path, "r") as f:
        return json.load(f)


def save_json(file, path):
    """
    Save a dictionary as a JSON file to a given path.

    Args:
    - file: dict - The dictionary to save as a JSON file
    - path: string - The path of the JSON file to save
    """
    with open(path, "w") as f:
        json.dump(file, f)


def mk_dir(path):
    """
    Create a directory if it does not already exist.

    Args:
    - path: string - The path of the directory to create
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def rm_spcl_char(text):
    """
    Remove special characters from a given text.

    Args:
    - text: string - The text from which to remove special characters

    Returns:
    - string - The text with all special characters removed
    """
    text = re.sub(
        r'[!@#$(),，\\n"%^*?？:;~`0-9&\\[\\]\\。\\/\\.\\=\\-]', " ", text
    )
    text = re.sub(r"[\\s]{2,}", " ", text.strip())
    text = text.lower().strip()
    return text


class ISO:
    """
    ISO Language Code Search and Conversion Tool

    This class provides a tool for searching and converting language codes from ISO 639-1 and 639-2 standards.

    """

    def __init__(self):
        """
        Initializes the ISO class by loading the JSON file into a dictionary,
        and creating a dictionary to map letters to language attributes.
        """
        self.dict_path = Path("resource/iso.json")
        self.iso_dict = load_json(str(self.dict_path))
        self.code_dict = dict(
            zip(
                ["e", "k", "1", "2"],
                ["English", "Korean", "ISO 639-1 Code", "ISO 639-2 Code"],
            )
        )

    def __call__(self, query, code=None, tol=2):
        """
        Searches the dictionary for entries matching the query and returns the results.

        Args:
        - query: string - The search query to match against the dictionary
        - code: string - A letter code representing an attribute of the language (e.g. 'e' for English)
        - tol: int - The tolerance level for matching. 0 for exact matches, 1 for case-insensitive, 2 for partial matches

        Returns:
        - list - A list of dictionary entries that match the search query and code (if specified)
        """
        return self._search(query, code, tol)

    def _search(self, query, code=None, tol=2):
        """
        Helper function to search the dictionary for matching entries.

        Args:
        - query: string - The search query to match against the dictionary
        - code: string - A letter code specifying the target cateogry  (e.g., 'e' for English, 'k' for Korean, '1' for ISO 639-1, and '2' for ISO 639-2)
        - tol: int - The tolerance level for matching. 0 for exact matches, 1 for case-insensitive, 2 for partial matches

        Returns:
        - list - A list of dictionary entries that match the search query and code (if specified)
        """
        results = []
        for row in self.iso_dict:
            if code:
                target = "%".join(
                    [
                        row.get(i)
                        for i in row
                        if i in [self.code_dict.get(c) for c in list(code)]
                    ]
                )
            else:
                target = "%".join(row.values())
            if self._word_validation(query, target, tol):
                results.append(row)
        return results

    def convert(self, src, src_code=None, dst_code=None):
        """
        Converts a language code from one standard to another.

        Args:
        - src: string - A source to convert
        - src_code: string - A letter code representing an attribute of the source (e.g. 'e' for English)
        - dst_code: string - A letter code representing the destination language to which the source convert (e.g. '1' for ISO 639-1)

        Returns:
        - string or list - A string or list of strings representing the converted language code(s)
        """
        search_result = self._search(src, code=src_code, tol=1)
        if not search_result:
            raise ValueError(
                "An input query can be case-insensitive but must be matched to a target completely."
            )
        result = search_result.pop(0)

        if dst_code:
            if len(dst_code) > 1:
                return [
                    result.get(self.code_dict.get(d)) for d in list(dst_code)
                ]
            return result.get(self.code_dict.get(dst_code))
        return result.values()

    def _word_validation(self, test: str, target: str, tol=2):
        """
        Helper function to validate matches based on the tolerance level.

        Args:
        - test: string - The search query to match against the target string
        - target: string - The string to match against the search query
        - tol: int - The tolerance level for matching. 0 for exact matches, 1 for case-insensitive, 2 for partial matches

        Returns:
        - bool - True if the search query matches the target string based on the tolerance level, False otherwise
        """
        if tol == 0:  # exact matches
            target_list = target.split("%")
            return test in target_list
        elif tol == 1:  # case-insensitive
            target_list = target.lower().split("%")
            return test.lower() in target_list
        elif tol == 2:  # partial matches
            return test.lower() in target.lower()
